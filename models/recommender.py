from abc import abstractmethod, ABC
import imp
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp.wrap import wrap
from torchmetrics import MetricCollection
from transformers import BertModel
from utils.metrics import MRR, NDCG, HR, get_topk_ranks
from utils.pylogger import get_pylogger
from utils.schedule_functions import get_lr_scheduler_function
from models.sasrec import SASRec
from models.opt import OPTModel
from models.layers import PromptEncoder, DeepPromptEncoder
from models.configs import (OPTSeqRecConfig, BERTSeqRecConfig, OPTPromptSeqRecConfig, 
                            BERTPromptSeqRecConfig, SeqRecConfig, TextSeqRecConfig)
from models.utils import gather_indexes, mean_pooling, last_pooling

log = get_pylogger(__name__)

METRIC_LIST = ["MRR", "NDCG", "HR"]


class SeqRec(pl.LightningModule, ABC):

    def __init__(self, config: SeqRecConfig):
        super().__init__()

        self.save_hyperparameters()

        self._set_feature_extractor(config)

        self.sasrec = SASRec(
            lr=config.lr,
            n_layers=config.sasrec_n_layers,
            n_heads=config.sasrec_n_heads,
            hidden_size=config.sasrec_hidden_size,
            inner_size=config.sasrec_inner_size,
            hidden_dropout=config.sasrec_hidden_dropout,
            attention_dropout=config.sasrec_attention_dropout,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            seq_len=config.sasrec_seq_len,
        )

        # project the last layer to the vocab size of item tokens
        hidden_size = self.sasrec.hidden_size
        item_token_num = config.item_token_num
        self.classification_head = nn.Linear(hidden_size, item_token_num)

        # metrics
        topk_list = config.topk_list
        self.topk_metric = {}
        self.topk_metric.update({f"MRR@{k}": MRR(k=k) for k in topk_list})
        self.topk_metric.update({f"HR@{k}": HR(k=k) for k in topk_list})
        self.topk_metric.update({f"NDCG@{k}": NDCG(k=k) for k in topk_list})
        self.topk_metric = MetricCollection(self.topk_metric)

        self.loss_fct = nn.CrossEntropyLoss()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            initializer_range = self.hparams.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    @abstractmethod
    def _set_feature_extractor(self):
        raise NotImplementedError

    @abstractmethod
    def _feature_extract(self, item_id_seq, item_seq_mask, tokenized_ids,
                         attention_mask):
        raise NotImplementedError

    def forward(self, item_id_seq, item_seq_mask, tokenized_ids,
                attention_mask):
        item_embs = self._feature_extract(item_id_seq, item_seq_mask,
                                          tokenized_ids, attention_mask)
        output = self.sasrec(item_embs, item_seq_mask)  # (B, L_sas, H_sas)
        output = self.classification_head(output)
        return output  # (B, L, N_items)

    def training_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, \
            tokenized_ids, attention_mask = batch
        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)  # (B, L, N_items)
        loss = self.loss_fct(seq_emb.reshape(-1, seq_emb.size(-1)),
                             target_id_seq.reshape(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, \
                     tokenized_ids, attention_mask = batch
        # (B, L, N_items)
        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)
        # (B)
        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1
        # (B, N_items)
        seq_last_emb = gather_indexes(seq_emb, last_item_idx)
        # (B, 1)
        last_item = target_id_seq.gather(1, last_item_idx.view(-1, 1))

        topk_list = self.hparams.config.topk_list
        pred_scores = seq_last_emb.softmax(dim=-1)
        all_ranks = get_topk_ranks(pred_scores=pred_scores,
                                   target=last_item,
                                   topk=max(topk_list))

        for k in topk_list:
            for metric_name in METRIC_LIST:
                metric = self.topk_metric[f"{metric_name}@{k}"]
                metric.update(all_ranks, last_item.numel())

    def validation_epoch_end(self, outputs):
        topk_list = self.hparams.config.topk_list
        for topk in topk_list:
            for metric_name in METRIC_LIST:
                score = self.topk_metric[f"{metric_name}@{topk}"].compute()
                if metric_name in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(f"val_{metric_name}@{topk}",
                         score,
                         on_epoch=True,
                         prog_bar=log_on_progress_bar,
                         logger=True,
                         sync_dist=True)

    def test_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, \
                     tokenized_ids, attention_mask = batch
        # (B, L, N_items)
        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)
        # (B)
        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1
        # (B, N_items)
        seq_last_emb = gather_indexes(seq_emb, last_item_idx)
        # (B, 1)
        last_item = target_id_seq.gather(1, last_item_idx.view(-1, 1))

        topk_list = self.hparams.config.topk_list
        pred_scores = seq_last_emb.softmax(dim=-1)
        all_ranks = get_topk_ranks(pred_scores=pred_scores,
                                   target=last_item,
                                   topk=max(topk_list))

        for k in topk_list:
            for metric_name in METRIC_LIST:
                metric = self.topk_metric[f"{metric_name}@{k}"]
                metric.update(all_ranks, last_item.numel())

    def test_epoch_end(self, outputs):
        topk_list = self.hparams.config.topk_list
        for topk in topk_list:
            for metric_name in METRIC_LIST:
                score = self.topk_metric[f"{metric_name}@{topk}"].compute()
                if metric_name in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(f"test_{metric_name}@{topk}",
                         score,
                         on_epoch=True,
                         prog_bar=log_on_progress_bar,
                         logger=True,
                         sync_dist=True)

    def configure_optimizers(self):
        lr = self.hparams.config.lr
        wd = self.hparams.config.weight_decay
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=lr,
                                      weight_decay=wd)
        return optimizer
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("SeqRec")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=0.1)
        parser.add_argument("--sasrec_n_layers", type=int, default=2)
        parser.add_argument("--sasrec_n_heads", type=int, default=2)
        parser.add_argument("--sasrec_hidden_size", type=int, default=64)
        parser.add_argument("--sasrec_inner_size", type=int, default=256)
        parser.add_argument("--sasrec_hidden_dropout", type=float, default=0.5)
        parser.add_argument("--sasrec_attention_dropout", type=float, default=0.5)
        parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--topk_list",
                            type=int,
                            nargs="+",
                            default=[5, 10, 20])
        return parent_parser
    
    @classmethod
    def build_model_config(cls, args, config):
        # shared parameters of SeqRec
        config.lr = args.lr
        config.weight_decay = args.weight_decay
        config.sasrec_seq_len = args.sasrec_seq_len
        config.sasrec_n_layers = args.sasrec_n_layers
        config.sasrec_n_heads = args.sasrec_n_heads
        config.sasrec_hidden_size = args.sasrec_hidden_size
        config.sasrec_inner_size = args.sasrec_inner_size
        config.sasrec_hidden_dropout = args.sasrec_hidden_dropout
        config.sasrec_attention_dropout = args.sasrec_attention_dropout
        config.layer_norm_eps = args.layer_norm_eps
        config.initializer_range = args.initializer_range
        config.topk_list = args.topk_list
        return config


class IDSeqRec(SeqRec):

    def __init__(self, config: SeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

        # parameters initialization
        self.apply(self._init_weights)

    def _set_feature_extractor(self, config):
        self.item_embedding = torch.nn.Embedding(config.item_token_num,
                                                 config.sasrec_hidden_size,
                                                 padding_idx=0)

    def _feature_extract(self, item_id_seq, item_seq_mask, tokenized_ids,
                         attention_mask):
        item_embs = self.item_embedding(item_id_seq)
        return item_embs

    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = SeqRecConfig(item_token_num=item_token_num)
        config = super(IDSeqRec, cls).build_model_config(args, config)
        return config

class TextSeqRec(SeqRec, ABC):

    def __init__(self, config: TextSeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

    def _set_feature_extractor(self, config):
        plm_name = config.plm_name
        num_unfreeze_layers = config.plm_n_unfreeze_layers
        self._set_plm_model(plm_name)
        self._freeze_plm_layers(num_unfreeze_layers)

        output_size = self._get_item_emb_dim()

        projection_n_layers = config.projection_n_layers
        projection_inner_sizes = config.projection_inner_sizes
        hidden_size = config.sasrec_hidden_size
        projection_sizes = [output_size] + \
            projection_inner_sizes + [hidden_size]
        # mlps with residual connections for projection
        self.projection = torch.nn.ModuleList()
        self.projection = nn.ModuleList()
        for i in range(projection_n_layers):
            self.projection.append(
                nn.Sequential(
                    nn.Linear(projection_sizes[i], projection_sizes[i + 1]),
                    # nn.GELU()
                ))
        # layer_norm_eps = config.layer_norm_eps
        # self.projection.append(nn.LayerNorm(hidden_size, eps=layer_norm_eps))

    @abstractmethod
    def _set_plm_model(self, config):
        raise NotImplementedError

    @abstractmethod
    def _freeze_plm_layers(self, num_unfreeze_layers):
        raise NotImplementedError

    @abstractmethod
    def _get_item_emb_dim(self):
        raise NotImplementedError

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super(TextSeqRec, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("TextSeqRec")
        # shared parameters of TextSeqRec
        parser.add_argument("--projection_n_layers", type=int, default=5)
        parser.add_argument("--projection_inner_sizes",
                            type=int,
                            nargs="*",
                            default=[3136, 784, 3136, 784])
        return parent_parser
    
    @classmethod
    def build_model_config(cls, args, config):
        config = super(TextSeqRec, cls).build_model_config(args, config)
        config.plm_name = args.plm_name
        return config

class OPTSeqRec(TextSeqRec):

    def __init__(self, config: OPTSeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

        # parameters initialization
        self.apply(self._init_weights)

    def _set_plm_model(self, plm_name):
        self.opt = OPTModel.from_pretrained(plm_name)

    def _get_item_emb_dim(self):
        return self.opt.config.hidden_size

    def _freeze_plm_layers(self, num_unfreeze_layers):
        plm_n_layers = self.opt.config.num_hidden_layers
        if num_unfreeze_layers < -1 or num_unfreeze_layers > plm_n_layers:
            raise ValueError(
                f"num_unfreeze_layers {num_unfreeze_layers} is not supported.")

        if num_unfreeze_layers == -1:
            for param in self.opt.parameters():
                param.requires_grad = True
        else:
            for param in self.opt.parameters():
                param.requires_grad = False

        if num_unfreeze_layers > 0:
            unfreeze_layers = self.opt.decoder.layers[-num_unfreeze_layers:]
            for param in unfreeze_layers.parameters():
                param.requires_grad = True

    def _feature_extract(self, item_id_seq, item_seq_mask, tokenized_ids,
                         attention_mask):
        tokenized_ids = tokenized_ids.view(-1, tokenized_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        item_embs = self._get_opt_output(tokenized_ids, attention_mask)

        for layer in self.projection:
            item_embs = layer(item_embs)

        sasrec_seq_len = self.hparams.config.sasrec_seq_len
        sasrec_hidden_size = self.hparams.config.sasrec_hidden_size
        item_embs = item_embs.view(-1, sasrec_seq_len, sasrec_hidden_size)
        return item_embs

    def _get_opt_output(self, tokenized_ids, attention_mask):
        if self.hparams.config.plm_n_unfreeze_layers == 0:
            with torch.no_grad():
                output = self.opt(input_ids=tokenized_ids,
                                  attention_mask=attention_mask)
        else:
            output = self.opt(input_ids=tokenized_ids,
                              attention_mask=attention_mask)
        # (B * L_sas, L_plm, H_plm)
        sentence_embs = output.last_hidden_state
        pooling_method = self.hparams.config.pooling_method
        if pooling_method == "mean":
            # (B * L_sas, H_plm)
            item_embs = mean_pooling(sentence_embs,
                                     attention_mask).type_as(sentence_embs)
        elif pooling_method == "last":
            # (B * L_sas, H_plm)
            item_embs = last_pooling(sentence_embs,
                                     attention_mask).type_as(sentence_embs)
        return item_embs

    def _set_opt_lr(self, lr, decay, wd):
        tuning_params = []
        n_layers = self.opt.config.num_hidden_layers
        lrs = [lr * (decay**(n_layers - i)) for i in range(n_layers)]
        no_weight_decay = ["bias", "LayerNorm.weight"]

        for name, params in self.opt.named_parameters():
            if name.startswith("decoder.layers"):
                layer_idx = int(name.split(".")[2])
                p = {"params": params, "lr": lrs[layer_idx], "name": name}
            elif name.startswith("decoder.embed_"):
                p = {"params": params, "lr": lrs[0], "name": name}
            else:
                p = {"params": params, "lr": lrs[-1], "name": name}
            if any(nd in name for nd in no_weight_decay):
                p.update(weight_decay=0.0)
            else:
                p.update(weight_decay=wd)
            tuning_params.append(p)

        tuning_params = [
            layer for layer in tuning_params if layer["params"].requires_grad
        ]
        return tuning_params

    def configure_optimizers(self):
        lr = self.hparams.config.lr
        wd = self.hparams.config.weight_decay
        if self.hparams.config.plm_n_unfreeze_layers == 0:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=lr,
                                          weight_decay=wd)
        else:
            plm_lr = self.hparams.config.plm_lr
            layer_decay = self.hparams.config.plm_lr_layer_decay
            # set different learning rate for different layers
            opt_tuning_params = self._set_opt_lr(plm_lr, layer_decay, wd)
            opt_tuning_names = [
                "opt." + layer["name"] for layer in opt_tuning_params
            ]
            the_rest_params = []
            for name, params in self.named_parameters():
                if name not in opt_tuning_names:
                    the_rest_params.append(params)
            the_rest_params = [{
                "params": the_rest_params,
                "lr": lr,
                "weight_decay": wd,
                "name": "the_rest"
            }]
            all_params = opt_tuning_params + the_rest_params
            optimizer = torch.optim.AdamW(all_params)
        return optimizer
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super(OPTSeqRec, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("OPTSeqRec")
        parser.add_argument("--plm_n_unfreeze_layers", type=int, default=0)
        # shared parameters of fine-tuneing PLM
        parser.add_argument("--plm_lr", type=float, default=1e-5)
        parser.add_argument("--plm_lr_layer_decay", type=float, default=0.8)
        parser.add_argument("--pooling_method", type=str, default="mean")
        return parent_parser

    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = OPTSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers=args.plm_n_unfreeze_layers,
                    plm_lr=args.plm_lr,
                    plm_lr_layer_decay=args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                )
        config = super(OPTSeqRec, cls).build_model_config(args, config)
        return config


class OPTPromptSeqRec(OPTSeqRec):

    def __init__(self, config: OPTPromptSeqRecConfig):
        self.save_hyperparameters()
        super(OPTSeqRec, self).__init__(self.hparams.config)

        if config.pre_seq_len > 0:
            self.prefix_encoder = DeepPromptEncoder(
                plm=self.opt,
                prompt_projection=config.prompt_projection,
                prompt_seq_len=config.pre_seq_len,
                prompt_hidden_size=config.prompt_hidden_size,
                layer_norm_eps=config.layer_norm_eps)

        if config.post_seq_len > 0:
            self.postfix_encoder = DeepPromptEncoder(
                plm=self.opt,
                prompt_projection=config.prompt_projection,
                prompt_seq_len=config.post_seq_len,
                prompt_hidden_size=config.prompt_hidden_size,
                layer_norm_eps=config.layer_norm_eps)
            assert config.last_query_len >= 1, \
                "last_query_len must be at least 1"

        if config.last_query_len > 0:
            self.last_query_encoder = PromptEncoder(
                plm=self.opt, prompt_seq_len=config.last_query_len)

        if config.pooling_method == "mean_last":
            plm_hidden_size = self.opt.config.hidden_size
            eps = config.layer_norm_eps
            self.fusion_mlp = torch.nn.Sequential(
                torch.nn.Linear(plm_hidden_size * 2, plm_hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(plm_hidden_size, plm_hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(plm_hidden_size, plm_hidden_size),
                torch.nn.LayerNorm(plm_hidden_size, eps=eps))

        # parameters initialization
        self.apply(self._init_weights)

    def _get_opt_output(self, tokenized_ids, attention_mask):
        pre_seq_len = self.hparams.config.pre_seq_len
        post_seq_len = self.hparams.config.post_seq_len
        last_query_len = self.hparams.config.last_query_len
        plm_batch_size = tokenized_ids.shape[0]

        if pre_seq_len > 0:
            past_key_values = self.prefix_encoder(plm_batch_size)
            prefix_attention_mask = torch.ones(
                plm_batch_size, pre_seq_len).type_as(attention_mask)
            prompt_attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)
            output = self.opt(
                input_ids=tokenized_ids,
                attention_mask=prompt_attention_mask,
                past_key_values=past_key_values,
            )
        else:
            output = self.opt(input_ids=tokenized_ids,
                              attention_mask=attention_mask)
            prompt_attention_mask = attention_mask
        past_key_values = output.past_key_values

        if last_query_len > 0:
            if post_seq_len > 0:
                prompt_key_values = self.postfix_encoder(plm_batch_size)
                new_past_key_values = []
                for past_key_value, prompt_key_value in zip(
                        past_key_values, prompt_key_values):
                    key_states = torch.cat(
                        (past_key_value[0], prompt_key_value[0]), dim=2)
                    values_states = torch.cat(
                        (past_key_value[1], prompt_key_value[1]), dim=2)
                    new_past_key_values.append((key_states, values_states))
                past_key_values = new_past_key_values

            post_fix_attention_mask = torch.ones(
                plm_batch_size,
                post_seq_len + last_query_len).type_as(attention_mask)
            prompt_attention_mask = torch.cat(
                (prompt_attention_mask, post_fix_attention_mask), dim=1)

            last_query_embs = self.last_query_encoder(plm_batch_size)
            last_token_embs = self.opt(
                inputs_embeds=last_query_embs,
                attention_mask=prompt_attention_mask,
                past_key_values=past_key_values,
            )

        sentence_embs = output.last_hidden_state  # (B * L_sas, L_plm, H_plm)
        pooling_method = self.hparams.config.pooling_method
        if pooling_method == "mean":
            # (B * L_sas, H_plm)
            item_embs = mean_pooling(sentence_embs, attention_mask)
        elif pooling_method == "last":
            # (B * L_sas, H_plm)
            item_embs = last_token_embs.last_hidden_state[:, -1, :]
        elif pooling_method == "mean_last":
            mean_embs = mean_pooling(sentence_embs, attention_mask)
            last_embs = last_token_embs.last_hidden_state[:, -1, :]
            item_embs = torch.cat([mean_embs, last_embs], dim=-1)
            # (B * L_sas, H_plm)
            item_embs = self.fusion_mlp(item_embs)
        return item_embs

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super(OPTPromptSeqRec, cls).add_model_specific_args(parent_parser)
        parser = parser.add_argument_group("OPTPromptSeqRec")
        parser.add_argument("--pre_seq_len", type=int, default=20)
        parser.add_argument("--post_seq_len", type=int, default=10)
        parser.add_argument("--last_query_len", type=int, default=1)
        parser.add_argument("--prompt_hidden_size", type=int, default=128)
        parser.add_argument("--prompt_projeciton",
                                type=str,
                                default="nonlinear")
        return parent_parser
    
    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = OPTPromptSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers=args.plm_n_unfreeze_layers,
                    plm_lr=args.plm_lr,
                    plm_lr_layer_decay=args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                    prompt_projection=args.prompt_projeciton,
                    prompt_hidden_size=args.prompt_hidden_size,
                    pre_seq_len=args.pre_seq_len,
                    post_seq_len=args.post_seq_len,
                    last_query_len=args.last_query_len,
                )
        config = super(OPTSeqRec, cls).build_model_config(args, config)
        return config

class BERTSeqRec(TextSeqRec):

    def __init__(self, config: BERTSeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

        # parameters initialization
        self.apply(self._init_weights)

    def _set_plm_model(self, plm_name):
        self.bert = BertModel.from_pretrained(plm_name)

    def _get_item_emb_dim(self):
        return self.bert.config.hidden_size

    def _freeze_plm_layers(self, num_unfreeze_layers):
        plm_n_layers = self.bert.config.num_hidden_layers
        if num_unfreeze_layers < -1 or num_unfreeze_layers > plm_n_layers:
            raise ValueError(
                f"num_unfreeze_layers {num_unfreeze_layers} is not supported.")

        if num_unfreeze_layers == -1:
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

        if num_unfreeze_layers > 0:
            unfreeze_layers = self.bert.encoder.layer[-num_unfreeze_layers:]
            for param in unfreeze_layers.parameters():
                param.requires_grad = True

    def _get_bert_output(self, tokenized_ids, attention_mask):
        if self.hparams.config.plm_n_unfreeze_layers == 0:
            with torch.no_grad():
                output = self.bert(input_ids=tokenized_ids,
                                   attention_mask=attention_mask)
        else:
            output = self.bert(input_ids=tokenized_ids,
                               attention_mask=attention_mask)

        pooling_method = self.hparams.config.pooling_method
        if pooling_method == "mean":
            # (B * L_sas, L_plm, H_plm)
            sentence_embs = output.last_hidden_state
            # (B * L_sas, H_plm)
            item_embs = mean_pooling(sentence_embs,
                                     attention_mask).type_as(sentence_embs)
        elif pooling_method == "cls":
            item_embs = output.last_hidden_state[:, 0, :]
        elif pooling_method == "pooler":
            item_embs = output.pooler_output  # (B * L_sas, H_plm)
        return item_embs

    def _feature_extract(self, item_id_seq, item_seq_mask, tokenized_ids,
                         attention_mask):

        tokenized_ids = tokenized_ids.view(-1, tokenized_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        item_embs = self._get_bert_output(tokenized_ids, attention_mask)

        for layer in self.projection:
            item_embs = layer(item_embs)

        sasrec_seq_len = self.hparams.config.sasrec_seq_len
        sasrec_hidden_size = self.hparams.config.sasrec_hidden_size
        item_embs = item_embs.view(-1, sasrec_seq_len, sasrec_hidden_size)
        return item_embs

    def _set_bert_lr(self, lr, decay, wd):
        tuning_params = []
        n_layers = self.bert.config.num_hidden_layers
        lrs = [lr * (decay**(n_layers - i)) for i in range(n_layers)]
        no_weight_decay = ["bias", "LayerNorm.weight"]

        for name, params in self.bert.named_parameters():
            if name.startswith("encoder.layer"):
                layer_idx = int(name.split(".")[2])
                p = {"params": params, "lr": lrs[layer_idx], "name": name}
            elif name.startswith("embeddings"):
                p = {"params": params, "lr": lrs[0], "name": name}
            else:
                p = {"params": params, "lr": lrs[-1], "name": name}
            if any(nd in name for nd in no_weight_decay):
                p.update(weight_decay=0.0)
            else:
                p.update(weight_decay=wd)
            tuning_params.append(p)

        tuning_params = [
            layer for layer in tuning_params if layer["params"].requires_grad
        ]
        return tuning_params

    def configure_optimizers(self):
        lr = self.hparams.config.lr
        wd = self.hparams.config.weight_decay
        if self.hparams.config.plm_n_unfreeze_layers == 0:
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=lr,
                                          weight_decay=wd)
        else:
            plm_lr = self.hparams.config.plm_lr
            layer_decay = self.hparams.config.plm_lr_layer_decay
            # set different learning rate for different layers
            bert_tuning_params = self._set_bert_lr(plm_lr, layer_decay, wd)
            bert_tuning_names = [
                "bert." + layer["name"] for layer in bert_tuning_params
            ]
            the_rest_layers = []
            for name, params in self.named_parameters():
                if name not in bert_tuning_names:
                    # if name.startswith("projection"):
                    #     the_rest_layers.append({"params": params, "lr": 1e-3, "name": name})
                    # else:
                    the_rest_layers.append({
                        "params": params,
                        "lr": lr,
                        "weight_decay": wd,
                        "name": name
                    })

            all_params = bert_tuning_params + the_rest_layers
            optimizer = torch.optim.AdamW(all_params)
            # warmup_type = "linear"
            # if warmup_type is not None:
            #     scheduler = []
            #     warmup_steps = self.hparams.config.warmup_steps
            #     total_steps = self.hparams.config.total_steps

            #     lr_lambda = [get_lr_scheduler_function(warmup_type, warmup_steps, total_steps)]
            #     scheduler.append({scheduler=LambdaLR(optimizer, lr_lambda=lr_lambda), interval="step"})
        return optimizer
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super(BERTSeqRec, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("BERTSeqRec")
        parser.add_argument("--plm_n_unfreeze_layers", type=int, default=0)
        # shared parameters of fine-tuneing PLM
        parser.add_argument("--plm_lr", type=float, default=1e-5)
        parser.add_argument("--plm_lr_layer_decay", type=float, default=0.8)
        parser.add_argument("--pooling_method", type=str, default="cls")
        return parent_parser

    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = BERTSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers=args.plm_n_unfreeze_layers,
                    plm_lr=args.plm_lr,
                    plm_lr_layer_decay=args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                )
        config = super(BERTSeqRec, cls).build_model_config(args, config)
        return config


class BERTPromptSeqRec(BERTSeqRec):

    def __init__(self, config: BERTPromptSeqRecConfig):
        self.save_hyperparameters()
        super(BERTSeqRec, self).__init__(self.hparams.config)

        self.prefix_encoder = DeepPromptEncoder(
            plm=self.bert,
            prompt_projection=config.prompt_projection,
            prompt_seq_len=config.pre_seq_len,
            prompt_hidden_size=config.prompt_hidden_size,
            layer_norm_eps=config.layer_norm_eps)

        # parameters initialization
        self.apply(self._init_weights)

    def _get_bert_output(self, tokenized_ids, attention_mask):
        pre_seq_len = self.hparams.config.pre_seq_len
        plm_batch_size = tokenized_ids.shape[0]

        past_key_values = self.prefix_encoder(plm_batch_size)
        prefix_attention_mask = torch.ones(plm_batch_size,
                                           pre_seq_len).type_as(attention_mask)
        prompt_attention_mask = torch.cat(
            (prefix_attention_mask, attention_mask), dim=1)
        output = self.bert(
            input_ids=tokenized_ids,
            attention_mask=prompt_attention_mask,
            past_key_values=past_key_values,
        )

        pooling_method = self.hparams.config.pooling_method
        if pooling_method == "mean":
            # (B * L_sas, L_plm, H_plm)
            sentence_embs = output.last_hidden_state
            # (B * L_sas, H_plm)
            item_embs = mean_pooling(sentence_embs, attention_mask)
        elif pooling_method == "cls":
            item_embs = output.last_hidden_state[:, 0, :]
        elif pooling_method == "pooler":
            item_embs = output.pooler_output  # (B * L_sas, H_plm)
        return item_embs
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super(BERTPromptSeqRec, cls).add_model_specific_args(parent_parser)
        parser = parser.add_argument_group("BERTPromptSeqRec")
        parser.add_argument("--pre_seq_len", type=int, default=20)
        parser.add_argument("--prompt_hidden_size", type=int, default=128)
        parser.add_argument("--prompt_projeciton",
                                type=str,
                                default="nonlinear")
        return parent_parser
    
    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = BERTPromptSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers=args.plm_n_unfreeze_layers,
                    plm_lr=args.plm_lr,
                    plm_lr_layer_decay=args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                    prompt_projection=args.prompt_projeciton,
                    prompt_hidden_size=args.prompt_hidden_size,
                    pre_seq_len=args.pre_seq_len,
                )
        config = super(BERTSeqRec, cls).build_model_config(args, config)
        return config

class PreInferOPTSeqRec(OPTSeqRec):
    def __init__(self, trainable_opt_part, tokenized_embs_lookup, config: OPTSeqRecConfig):
        self.opt = trainable_opt_part
        self.tokenized_embs_lookup = tokenized_embs_lookup
        
        # normalized the tokenized_embs_lookup
        self.tokenized_embs_lookup = self.tokenized_embs_lookup / torch.norm(self.tokenized_embs_lookup, dim=1, keepdim=True)

        # transform the tokenized_embs_lookup to an embedding layer
        self.tokenized_embs_lookup = nn.Embedding.from_pretrained(self.tokenized_embs_lookup)
        self.tokenized_embs_lookup.weight.requires_grad = False
        
        self.save_hyperparameters(ignore=["trainable_opt_part", "tokenized_embs_lookup"])
        super(PreInferOPTSeqRec, self).__init__(self.hparams.config)
    
    def _set_plm_model(self, plm_name):
        pass
    
    def _freeze_plm_layers(self, num_unfreeze_layers):
        pass

    def _get_opt_output(self, tokenized_ids, attention_mask):
        if self.hparams.config.plm_n_unfreeze_layers == 0:
            tokenized_embs = self.tokenized_embs_lookup(tokenized_ids)
        else:
            tokenized_embs = self.tokenized_embs_lookup(tokenized_ids)
            output = self.opt(input_embs=tokenized_embs,
                              attention_mask=attention_mask)
            
        # (B * L_sas, L_plm, H_plm)
        sentence_embs = output.last_hidden_state
        pooling_method = self.hparams.config.pooling_method
        if pooling_method == "mean":
            # (B * L_sas, H_plm)
            item_embs = mean_pooling(sentence_embs,
                                     attention_mask).type_as(sentence_embs)
        elif pooling_method == "last":
            # (B * L_sas, H_plm)
            item_embs = last_pooling(sentence_embs,
                                     attention_mask).type_as(sentence_embs)
        return item_embs
    
    