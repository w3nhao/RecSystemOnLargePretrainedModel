import torch
from transformers import BertModel
from utils.pylogger import get_pylogger
from utils.metrics import get_topk_ranks
from utils.schedule_functions import get_lr_scheduler_function
from models.layers import DeepPromptEncoder
from models.abstract_recommender import TextSeqRec, METRIC_LIST

from models.configs import BERTSeqRecConfig, BERTPromptSeqRecConfig
from models.utils import mean_pooling, gather_indexes

log = get_pylogger(__name__)


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

    def _feature_extract(self, input_ids, attention_mask):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        item_embs = self._get_bert_output(input_ids, attention_mask)

        for layer in self.projection:
            item_embs = layer(item_embs)

        sasrec_seq_len = self.hparams.config.sasrec_seq_len
        sasrec_hidden_size = self.hparams.config.sasrec_hidden_size
        item_embs = item_embs.view(-1, sasrec_seq_len, sasrec_hidden_size)
        return item_embs

    def forward(self, item_seq_mask, input_ids, attention_mask):
        item_embs = self._feature_extract(input_ids, attention_mask)
        output = self.sasrec(item_embs, item_seq_mask)  # (B, L_sas, H_sas)
        output = self.classification_head(output)
        return output  # (B, L, N_items)
    
    def training_step(self, batch, batch_idx):
        target_seq, _, item_seq_mask, input_ids, attention_mask = batch
        seq_emb = self.forward(
            item_seq_mask, input_ids, attention_mask)  # (B, L, N_items)
        loss = self.loss_fct(seq_emb.reshape(-1, seq_emb.size(-1)),
                             target_seq.reshape(-1))
        return loss
    
    def _val_test_step(self, batch, batch_idx, stage):
        target_seq, _, item_seq_mask, input_ids, attention_mask = batch
        
        seq_emb = self.forward(
            item_seq_mask, input_ids, attention_mask) # (B, L, N_items)
        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1 # (B)
        seq_last_emb = gather_indexes(seq_emb, last_item_idx) # (B, N_items)
        last_id = target_seq.gather(1, last_item_idx.view(-1, 1)) # (B, 1)

        topk_list = self.hparams.config.topk_list
        pred_scores = seq_last_emb.softmax(dim=-1)
        all_ranks = get_topk_ranks(pred_scores=pred_scores,
                                   target=last_id,
                                   topk=max(topk_list))

        for k in topk_list:
            for metric_name in METRIC_LIST:
                metric = self.topk_metric[f"{metric_name}@{k}"]
                metric.update(all_ranks, last_id.numel())

    def _get_bert_output(self, input_ids, attention_mask):
        if self.hparams.config.plm_n_unfreeze_layers == 0:
            with torch.no_grad():
                output = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask)
        else:
            output = self.bert(input_ids=input_ids,
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

    def _feature_extract(self, item_id_seq, item_seq_mask, input_ids,
                         attention_mask):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        item_embs = self._get_bert_output(input_ids, attention_mask)

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

    def _get_bert_output(self, input_ids, attention_mask):
        pre_seq_len = self.hparams.config.pre_seq_len
        plm_batch_size = input_ids.shape[0]

        past_key_values = self.prefix_encoder(plm_batch_size)
        prefix_attention_mask = torch.ones(plm_batch_size,
                                           pre_seq_len).type_as(attention_mask)
        prompt_attention_mask = torch.cat(
            (prefix_attention_mask, attention_mask), dim=1)
        output = self.bert(
            input_ids=input_ids,
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
        parser = super(BERTPromptSeqRec,
                       cls).add_model_specific_args(parent_parser)
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
