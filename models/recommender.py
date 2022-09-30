import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from models.sasrec import SASRec
from models.layers import PromptEncoder
from transformers import BertModel
from models.utils import get_plm_configs, get_plm_model, freeze_plm_layers
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
        prompt_projection,
        prompt_hidden_size,
        pre_seq_len,
        post_seq_len,
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
        use_pre_prompt=False,
        use_post_prompt=False,
        use_mlp_projection=False,
        top_k_list=[5, 10, 20],
    ):
        super(SeqRecommender, self).__init__()

        self.item_token_num = item_token_num
        self.sasrec_seq_len = sasrec_seq_len

        self.input_type = input_type
        self.plm = plm
        self.num_unfreeze_layers = num_unfreeze_layers

        self.use_pre_prompt = use_pre_prompt
        self.use_post_prompt = use_post_prompt
        self.prompt_projection = prompt_projection
        self.pre_seq_len = pre_seq_len
        self.post_seq_len = post_seq_len
        self.prompt_hidden_size = prompt_hidden_size
        self.pooling_type = pooling_type
        self.use_mlp_projection = use_mlp_projection

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

        ignore_args = []
        if input_type == "id":
            self.item_embedding = torch.nn.Embedding(item_token_num,
                                                     hidden_size,
                                                     padding_idx=0)
            ignore_args.append(["plm", "num_unfreeze_layers"])

        elif input_type == "text":
            self.item_feature_extractor = get_plm_model(plm)

            self.item_feature_extractor = freeze_plm_layers(
                self.item_feature_extractor, num_unfreeze_layers)

            output_size, plm_hidden_size, plm_n_layers, \
            _, _, _ = get_plm_configs(self.item_feature_extractor)

            if use_pre_prompt or use_post_prompt:
                if use_pre_prompt:
                    self.prefix_encoder = PromptEncoder(
                        plm=self.item_feature_extractor,
                        prompt_projection=prompt_projection,
                        prompt_seq_len=pre_seq_len,
                        num_hidden_layers=plm_n_layers,
                        hidden_size=plm_hidden_size,
                        prompt_hidden_size=prompt_hidden_size,
                        layer_norm_eps=layer_norm_eps
                    )

                if use_post_prompt:
                    if isinstance(self.item_feature_extractor, BertModel):
                        raise NotImplementedError(
                            "post prompt is not supported for bert.")

                    self.postfix_encoder = PromptEncoder(
                        plm=self.item_feature_extractor,
                        prompt_projection=prompt_projection,
                        prompt_seq_len=post_seq_len,
                        num_hidden_layers=plm_n_layers,
                        hidden_size=plm_hidden_size,
                        prompt_hidden_size=prompt_hidden_size,
                        layer_norm_eps=layer_norm_eps
                    )
                    self.last_query_embed = torch.nn.Parameter(
                        torch.randn(1, 1, plm_hidden_size))
                self.after_pooling_layernorm = torch.nn.LayerNorm(plm_hidden_size, eps=layer_norm_eps)
                
            if use_mlp_projection:
                mlp_sizes = [output_size] + mlp_inner_size + [hidden_size]
                assert mlp_layers_num > 2
                assert len(mlp_sizes) == mlp_layers_num + 1

                self.mlp_projection = nn.Sequential([
                    nn.Sequential(
                        nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]),
                        nn.ReLU(),
                        nn.LayerNorm(mlp_sizes[i + 1], eps=layer_norm_eps),
                    ) for i in range(mlp_layers_num)
                ])

                self.mlp_layers_num = mlp_layers_num
                self.mlp_inner_size = mlp_inner_size
            else:
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

            output = None
            if self.use_pre_prompt:
                past_key_values = self.prefix_encoder(plm_batch_size)
                prefix_attention_mask = torch.ones(
                    plm_batch_size, self.pre_seq_len).type_as(attention_mask)
                prompt_attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=1)
                output = self.item_feature_extractor(
                    input_ids=tokenized_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=past_key_values,
                )

            if self.use_post_prompt:
                if output is None:
                    output = self.item_feature_extractor(
                        input_ids=tokenized_ids, attention_mask=attention_mask)
                    prompt_attention_mask = attention_mask
                past_key_values = output.past_key_values
                
                prompt_key_values = self.postfix_encoder(plm_batch_size)
                new_past_key_values = []
                for past_key_value, prompt_key_value in zip(
                        past_key_values, prompt_key_values):
                    key_states = torch.cat(
                        (past_key_value[0], prompt_key_value[0]), dim=2)
                    values_states = torch.cat(
                        (past_key_value[1], prompt_key_value[1]), dim=2)
                    new_past_key_values.append((key_states, values_states))

                post_fix_attention_mask = torch.ones(plm_batch_size,
                                                     self.post_seq_len +
                                                     1).type_as(attention_mask)
                prompt_attention_mask = torch.cat(
                    (prompt_attention_mask, post_fix_attention_mask), dim=1)

                last_query_embeds = self.last_query_embed.expand(
                    plm_batch_size, -1, -1)
                last_query = self.item_feature_extractor(
                    inputs_embeds=last_query_embeds,
                    attention_mask=prompt_attention_mask,
                    past_key_values=new_past_key_values,
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
                if not self.use_post_prompt:
                    # last pooling
                    last_token_idx = attention_mask.sum(dim=1) - 1
                    cumsum_idx = torch.tensor(sentence_len).expand(
                        plm_batch_size).cumsum(0).type_as(tokenized_ids)
                    last_token_idx[1:] += cumsum_idx[:-1]
                    item_embs = sentence_embs.view(
                        -1, sentence_embs.size(-1))[last_token_idx]
                else:
                    last_query_embs = last_query.last_hidden_state
                    item_embs = last_query_embs[:, 0, :]

            elif self.pooling_type == "mean_last":
                # mean pooling
                attn_mask_expanded = attention_mask.unsqueeze(-1).expand_as(
                    sentence_embs)
                sum_embs = (sentence_embs * attn_mask_expanded).sum(1)
                num_mask = torch.clamp(attn_mask_expanded.sum(1), min=1e-9)
                mean_item_embs = sum_embs / num_mask  # (B * L_sas, H_plm)
                # last pooling
                last_token_idx = attention_mask.sum(dim=1) - 1
                cumsum_idx = torch.tensor(sentence_len).expand(
                    plm_batch_size).cumsum(0).type_as(tokenized_ids)
                last_token_idx[1:] += cumsum_idx[:-1]
                last_item_embs = sentence_embs.view(
                    -1, sentence_embs.size(-1))[last_token_idx]
                item_embs = torch.cat((mean_item_embs, last_item_embs), dim=1)

            else:
                raise ValueError(
                    "pooling_type should be one of mean, last, mean_last")

            item_embs = self.after_pooling_layernorm(item_embs)

            if self.use_mlp_projection:
                item_embs = self.mlp_projection(item_embs)
            else:
                item_embs = self.linear_projection(
                    item_embs)  # (B * L_sas, H_sas)

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
                self.log(f"val_{metric_name}@{topk}",
                         score,
                         on_epoch=True,
                         prog_bar=log_on_progress_bar,
                         logger=True,
                         sync_dist=True)

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
                self.log(f"test_{metric_name}@{topk}",
                         score,
                         on_epoch=True,
                         prog_bar=log_on_progress_bar,
                         logger=True,
                         sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
