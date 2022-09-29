from gc import unfreeze
from importlib.util import module_for_loader
from lib2to3.pgen2 import token
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from models.sasrec import SASRec
from models.opt import OPTModel
from models.layers import PrefixEncoder
from transformers import BertModel
from utils.pylogger import get_pylogger
from utils.metrics import MRR, NDCG, HR, get_topk_ranks

log = get_pylogger(__name__)

METRIC_LIST = ["MRR", "NDCG", "HR"]


class SeqRecommender(pl.LightningModule):

    def __init__(
        self,
        item_token_num,
        sasrec_seq_len,
        input_type,
        plm,
        prefix_projection,
        prefix_hidden_size,
        pre_seq_len,
        num_unfreeze_layers,
        lr,
        n_layers,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout,
        attention_dropout,
        layer_norm_eps,
        initializer_range,
        mlp_layers_num,
        mlp_inner_size,
        pooling_type="mean",
        use_deep_prompt=False,
        use_mlp_projection=False,
        top_k_list=[5, 10, 20],
    ):
        super(SeqRecommender, self).__init__()
        ignore_args = []
        if input_type == "id":
            ignore_args.append(["plm", "num_unfreeze_layers"])
            self.item_feature_extractor = None
            self.item_embedding = torch.nn.Embedding(item_token_num,
                                                     hidden_size,
                                                     padding_idx=0)

        elif input_type == "text":
            self.item_embedding = None

            # TODO add BERT, RoBERTa, etc.
            if plm.startswith("facebook/opt"):
                self.item_feature_extractor = OPTModel.from_pretrained(
                    plm, output_hidden_states=True)
            elif plm.startswith("bert"):
                self.item_feature_extractor = BertModel.from_pretrained(
                    plm, output_hidden_states=True)
            else:
                raise ValueError(
                    f"plm {plm} is not supported.")

            output_size = self.item_feature_extractor.config.word_embed_proj_dim
            plm_hidden_size = self.item_feature_extractor.config.hidden_size
            plm_n_layers = self.item_feature_extractor.config.num_hidden_layers
            plm_n_heads = self.item_feature_extractor.config.num_attention_heads
            plm_n_embd = output_size // plm_n_heads
            
            if num_unfreeze_layers < 0 or num_unfreeze_layers > plm_n_layers:
                raise ValueError(
                    f"num_unfreeze_layers {num_unfreeze_layers} is not supported."
                )

            for param in self.item_feature_extractor.parameters():
                param.requires_grad = False

            if num_unfreeze_layers > 0:
                if isinstance(self.item_feature_extractor, OPTModel):
                    unfreeze_layers = self.item_feature_extractor.decoder.layers[
                        -num_unfreeze_layers:]
                elif isinstance(self.item_feature_extractor, BertModel):
                    unfreeze_layers = self.item_feature_extractor.encoder.layer[
                        -num_unfreeze_layers:]
                for param in unfreeze_layers.parameters():
                    param.requires_grad = True

            if use_deep_prompt:
                self.prefix_encoder = PrefixEncoder(
                    prefix_projection=prefix_projection,
                    pre_seq_len=pre_seq_len,
                    num_hidden_layers=plm_n_layers,
                    hidden_size=plm_hidden_size,
                    prefix_hidden_size=prefix_hidden_size,
                )
                self.register_buffer("prefix_tokens",
                                     torch.arange(pre_seq_len).long())

                if isinstance(self.item_feature_extractor, OPTModel):
                    plm_dropout = self.item_feature_extractor.config.dropout
                elif isinstance(self.item_feature_extractor, BertModel):
                    plm_dropout = self.item_feature_extractor.config.hidden_dropout_prob
                self.prompt_dropout = torch.nn.Dropout(
                    plm_dropout)

                self.prefix_projection = prefix_projection
                self.prefix_hidden_size = prefix_hidden_size
                self.plm_n_layers = plm_n_layers
                self.pre_seq_len = pre_seq_len
                self.plm_hidden_size = plm_hidden_size
                self.plm_n_heads = plm_n_heads
                self.plm_n_embd = plm_n_embd
                
            else:
                self.prefix_encoder = None

            if use_mlp_projection:
                self.linear_projection = None

                mlp_sizes = [output_size] + mlp_inner_size + [hidden_size]
                assert mlp_layers_num > 2
                assert len(mlp_sizes) == mlp_layers_num + 1

                self.mlp_projection = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]),
                        nn.ReLU(),
                        nn.LayerNorm(mlp_sizes[i + 1], eps=layer_norm_eps),
                    ) for i in range(mlp_layers_num)
                ])

                self.mlp_layers_num = mlp_layers_num
                self.mlp_inner_size = mlp_inner_size
            else:
                self.mlp_projection = None
                ignore_args.append(["mlp_layers_num", "mlp_inner_size"])
                self.linear_projection = nn.Linear(output_size, hidden_size)

        else:
            raise NotImplementedError("input_type must be 'id' or 'text'")

        self.sasrec = SASRec(
            lr=lr,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            inner_size=inner_size,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            initializer_range=initializer_range,
            seq_len=sasrec_seq_len,
        )

        # project the last layer to the vocab size of item tokens
        self.classification_head = nn.Linear(hidden_size, item_token_num)

        # metrics
        self.topk_metric = {}
        self.topk_metric.update({f"MRR@{k}": MRR(k=k) for k in top_k_list})
        self.topk_metric.update({f"HR@{k}": HR(k=k) for k in top_k_list})
        self.topk_metric.update({f"NDCG@{k}": NDCG(k=k) for k in top_k_list})
        self.topk_metric = MetricCollection(self.topk_metric)

        self.loss_fct = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=ignore_args)

        self.item_token_num = item_token_num
        self.sasrec_seq_len = sasrec_seq_len
        
        self.input_type = input_type
        self.plm = plm
        self.num_unfreeze_layers = num_unfreeze_layers
        
        self.use_deep_prompt = use_deep_prompt
        self.use_mlp_projection = use_mlp_projection
        self.pooling_type = pooling_type
        
        self.lr = lr
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        self.topk_list = top_k_list

        # parameters initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version 
            # which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1,
                                         1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def _get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(batch_size, self.pre_seq_len, self.plm_n_layers * 2,
                                               self.plm_n_heads, self.plm_n_embd)
        past_key_values = self.prompt_dropout(past_key_values)
        past_key_values = past_key_values.permute(2, 0, 3, 1, 4).split(2)
        return past_key_values

    def forward(self, item_id_seq, item_seq_mask, tokenized_ids,
                attention_mask):
        if self.input_type == "id":
            item_embs = self.item_embedding(item_id_seq)

        elif self.input_type == "text":
            # (B, L_sas, L_plm)
            tokenized_ids = tokenized_ids.view(-1, tokenized_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            plm_batch_size = tokenized_ids.shape[0]
            sentence_len = tokenized_ids.shape[1]

            if self.use_deep_prompt:
                past_key_values = self._get_prompt(batch_size=plm_batch_size)
                prefix_attention_mask = torch.ones(
                    plm_batch_size, self.pre_seq_len).type_as(attention_mask)
                prompt_attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=1)
                output = self.item_feature_extractor(
                    input_ids=tokenized_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=past_key_values,
                )
            else:
                output = self.item_feature_extractor(
                    input_ids=tokenized_ids, attention_mask=attention_mask)

            sentence_embs = output.last_hidden_state  # (B * L_sas, L_plm, H_plm)
            
            if self.pooling_type == "mean":
                # mean pooling
                attn_mask_expanded = attention_mask.unsqueeze(-1).expand_as(
                    sentence_embs)
                sum_embs = (sentence_embs * attn_mask_expanded).sum(1)
                num_mask = torch.clamp(attn_mask_expanded.sum(1), min=1e-9)
                item_embs = sum_embs / num_mask  # (B * L_sas, H_plm)
            elif self.pooling_type == "last":
                # last pooling
                last_token_idx = attention_mask.sum(dim=1) - 1
                cumsum_idx = torch.tensor(sentence_len).expand(plm_batch_size).cumsum(0).type_as(tokenized_ids)
                last_token_idx[1:] += cumsum_idx[:-1]
                item_embs = sentence_embs.view(-1, sentence_embs.size(-1))[last_token_idx]
            elif self.pooling_type == "mean_last":
                # mean pooling
                attn_mask_expanded = attention_mask.unsqueeze(-1).expand_as(
                    sentence_embs)
                sum_embs = (sentence_embs * attn_mask_expanded).sum(1)
                num_mask = torch.clamp(attn_mask_expanded.sum(1), min=1e-9)
                mean_item_embs = sum_embs / num_mask  # (B * L_sas, H_plm)
                # last pooling
                last_token_idx = attention_mask.sum(dim=1) - 1
                cumsum_idx = torch.tensor(sentence_len).expand(plm_batch_size).cumsum(0).type_as(tokenized_ids)
                last_token_idx[1:] += cumsum_idx[:-1]
                last_item_embs = sentence_embs.view(-1, sentence_embs.size(-1))[last_token_idx]
                item_embs = torch.cat((mean_item_embs, last_item_embs), dim=1)
            else:
                raise ValueError("pooling_type should be one of mean, last, mean_last")

            if self.use_mlp_projection:
                item_embs = self.mlp_projection(item_embs)
            else:
                item_embs = self.linear_projection(item_embs)  # (B * L_sas, H_sas)

            item_embs = item_embs.view(-1, self.sasrec_seq_len,
                                       self.hidden_size)

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

        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)  # (B, L, N_items)

        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1  # (B)
        seq_last_emb = self._gather_indexes(seq_emb,
                                            last_item_idx)  # (B, N_items)
        last_item = target_id_seq. \
            gather(1, last_item_idx.view(-1, 1))  # (B, 1)

        pred_scores = seq_last_emb.softmax(dim=-1)
        all_ranks = get_topk_ranks(pred_scores=pred_scores,
                                   target=last_item,
                                   topk=max(self.topk_list))

        for k in self.topk_list:
            for metric_name in METRIC_LIST:
                metric = self.topk_metric[f"{metric_name}@{k}"]
                metric.update(all_ranks, last_item.numel())

    def validation_epoch_end(self, outputs):
        for topk in self.topk_list:
            for metric_name in METRIC_LIST:
                score = self.topk_metric[f"{metric_name}@{topk}"].compute()
                if metric_name in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(
                    f"val_{metric_name}@{topk}",
                    score,
                    on_epoch=True,
                    prog_bar=log_on_progress_bar,
                    logger=True,
                )

    def test_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, \
                     tokenized_ids, attention_mask = batch

        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)  # (B, L, N_items)

        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1  # (B)
        seq_last_emb = self._gather_indexes(seq_emb,
                                            last_item_idx)  # (B, N_items)
        last_item = target_id_seq. \
            gather(1, last_item_idx.view(-1, 1))  # (B, 1)

        pred_scores = seq_last_emb.softmax(dim=-1)
        all_ranks = get_topk_ranks(pred_scores=pred_scores,
                                   target=last_item,
                                   topk=max(self.topk_list))

        for k in self.topk_list:
            for metric_name in METRIC_LIST:
                metric = self.topk_metric[f"{metric_name}@{k}"]
                metric.update(all_ranks, last_item.numel())

    def test_epoch_end(self, outputs):
        for topk in self.topk_list:
            for metric_name in METRIC_LIST:
                score = self.topk_metric[f"{metric_name}@{topk}"].compute()
                if metric_name in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(
                    f"test_{metric_name}@{topk}",
                    score,
                    on_epoch=True,
                    prog_bar=log_on_progress_bar,
                    logger=True,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
