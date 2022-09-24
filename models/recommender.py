import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import (
    RetrievalHitRate,
    MetricCollection,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)
from models.sasrec import SASRec
from models.opt import OPTModel

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)

METRIC_ABBR = {
    "RetrievalHitRate": "HR",
    "RetrievalNormalizedDCG": "NDCG",
    "RetrievalMRR": "MRR",
}


class SeqRecommender(pl.LightningModule):

    def __init__(
        self,
        item_token_num,
        sasrec_seq_len,
        input_type,
        pretrained_model,
        num_unfreeze_layers,
        no_grad,
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
        use_mlp_connect=False,
        top_k_list=[5, 10, 20],
    ):
        super(SeqRecommender, self).__init__()
        ignore_save_args = []
        if input_type == "id":
            ignore_save_args.append(
                ["pretrained_model", "num_unfreeze_layers", "no_grad"])

            self.item_embedding = torch.nn.Embedding(item_token_num,
                                                     hidden_size,
                                                     padding_idx=0)
            self.item_feature_extractor = None
            # if use_mlp_connect:
            #     raise NotImplementedError(
            #         "use_mlp_connect=True is not implemented for input_type=id"
            #     )

        elif input_type == "text":
            self.item_embedding = None

            # TODO add BERT, RoBERTa, etc.
            if pretrained_model.startswith("facebook/opt"):
                if num_unfreeze_layers == 0:
                    self.item_feature_extractor = OPTModel.from_pretrained(
                        pretrained_model, output_hidden_states=True)
                    for param in self.item_feature_extractor.parameters():
                        param.requires_grad = False

                elif num_unfreeze_layers > 0:
                    self.item_feature_extractor = OPTModel.from_pretrained(
                        pretrained_model, output_hidden_states=True)
                    for param in self.item_feature_extractor.parameters():
                        param.requires_grad = False
                    for param in self.item_feature_extractor.decoder.layers[
                            -num_unfreeze_layers:].parameters():
                        param.requires_grad = True

                else:
                    raise ValueError(
                        "num_unfreeze_layers should be non-negative integer.")

            output_size = self.item_feature_extractor.config.hidden_size

            if use_mlp_connect:
                assert mlp_layers_num > 2

                mlp_sizes = [output_size] + mlp_inner_size + [hidden_size]
                assert len(mlp_sizes) == mlp_layers_num + 1

                self.mlp_connector = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]),
                        nn.ReLU(),
                        nn.LayerNorm(mlp_sizes[i + 1], eps=layer_norm_eps),
                    ) for i in range(mlp_layers_num)
                ])
                self.linear_connector = None

            else:
                ignore_save_args.append(["mlp_layers_num", "mlp_inner_size"])
                self.mlp_connector = None
                self.linear_connector = nn.Linear(output_size, hidden_size)

        else:
            raise NotImplementedError("input_type must be 'id' or 'text'")

        self.save_hyperparameters(ignore=ignore_save_args)

        self.item_token_num = item_token_num
        self.sasrec_seq_len = sasrec_seq_len
        self.input_type = input_type
        self.pretrained_model = pretrained_model
        self.num_unfreeze_layers = num_unfreeze_layers
        self.no_grad = no_grad
        self.lr = lr
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = inner_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.mlp_layers_num = mlp_layers_num
        self.mlp_inner_size = mlp_inner_size
        self.use_mlp_connect = use_mlp_connect
        self.topk_list = top_k_list

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
        self.project_layer = nn.Linear(self.hidden_size, item_token_num)

        # metrics
        self.topk_metric = {}

        for topk in self.topk_list:
            metrics = MetricCollection([
                RetrievalHitRate(k=topk),
                RetrievalNormalizedDCG(k=topk),
                RetrievalMRR(k=topk),
            ])
            self.topk_metric[topk] = metrics

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
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
            # (B * L_sas, L_opt)
            tokenized_ids = tokenized_ids.view(-1, tokenized_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

            # When num_unfreeze_layers == 0, freeze all the parameters of OPT
            # but compute gradients of the extra layers. When num_unfreeze_layers > 0
            # unfreeze the last num_unfreeze_layers of OPT
            if self.num_unfreeze_layers >= 0 and not self.no_grad:
                output = self.item_feature_extractor(
                    input_ids=tokenized_ids, attention_mask=attention_mask)

            # freeze all the parameters of OPT and do not compute gradients
            elif self.num_unfreeze_layers == 0 and self.no_grad:
                with torch.no_grad():
                    output = self.item_feature_extractor(
                        input_ids=tokenized_ids, attention_mask=attention_mask)

            elif self.num_unfreeze_layers > 0 and self.no_grad:
                raise NotImplementedError(
                    "is_frozen=False and no_grad=True is not implemented"
                    "This means that the parameters of OPT are not fully frozen but"
                    "the gradients are still not computed. This is not supported."
                )

            else:
                raise NotImplementedError

            text_seq_embs = output.last_hidden_state  # (B * L_sas, L_opt, H_opt)
            attn_mask_expanded = attention_mask.unsqueeze(-1).expand_as(
                text_seq_embs)

            # mean pooling
            sum_embs = (text_seq_embs * attn_mask_expanded).sum(1)
            num_mask = torch.clamp(attn_mask_expanded.sum(1), min=1e-9)
            item_embs = sum_embs / num_mask  # (B * L_sas, H_opt)

            item_embs = self.linear_connector(item_embs)  # (B * L_sas, H_sas)
            item_embs = item_embs.view(-1, self.sasrec_seq_len,
                                       self.hidden_size)

        output = self.sasrec(item_embs, item_seq_mask)  # (B, L_sas, H_sas)
        output = self.project_layer(output)

        return output  # (B, L, N_items)

    def training_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, tokenized_ids, attention_mask = batch
        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)  # (B, L, N_items)
        loss = self.loss_fct(seq_emb.reshape(-1, seq_emb.size(-1)),
                             target_id_seq.reshape(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, tokenized_ids, attention_mask = batch

        # (B, L, N_items)
        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)

        # (B)
        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1

        # (B, N_items)
        seq_last_emb = self._gather_indexes(seq_emb, last_item_idx)

        # (B, 1)
        last_item = target_id_seq.gather(1, last_item_idx.view(-1, 1))

        preds_scores, preds_items = torch.topk(seq_last_emb.softmax(dim=-1),
                                               k=max(self.topk_list),
                                               dim=-1)

        query_indexes = (torch.arange(preds_scores.size(0)).reshape(
            -1, 1).expand(-1, preds_scores.size(1)).type_as(preds_items))
        metric_target = preds_items == last_item

        metric = {}
        for topk in self.topk_list:
            metric[topk] = self.topk_metric[topk](preds_scores, metric_target,
                                                  query_indexes)
        return metric

    def validation_epoch_end(self, outputs):
        metric = {}
        for topk in self.topk_list:
            metric[topk] = {}
            for metric_name in outputs[0][topk].keys():
                scores = [m[topk][metric_name] for m in outputs]
                metric[topk][metric_name] = torch.stack(scores).mean()
                if METRIC_ABBR[metric_name] in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(
                    f"val_{METRIC_ABBR[metric_name]}@{topk}",
                    metric[topk][metric_name],
                    on_epoch=True,
                    prog_bar=log_on_progress_bar,
                    logger=True,
                )

    def test_step(self, batch, batch_idx):
        item_id_seq, target_id_seq, item_seq_mask, tokenized_ids, attention_mask = batch

        # (B, L, N_items)
        seq_emb = self.forward(item_id_seq, item_seq_mask, tokenized_ids,
                               attention_mask)

        # (B)
        last_item_idx = torch.sum(item_seq_mask, dim=-1) - 1

        # (B, N_items)
        seq_last_emb = self._gather_indexes(seq_emb, last_item_idx)

        # (B, 1)
        last_item = target_id_seq.gather(1, last_item_idx.view(-1, 1))

        preds_scores, preds_items = torch.topk(seq_last_emb.softmax(dim=-1),
                                               k=max(self.topk_list),
                                               dim=-1)

        query_indexes = (torch.arange(preds_scores.size(0)).reshape(
            -1, 1).expand(-1, preds_scores.size(1)).type_as(preds_items))
        metric_target = preds_items == last_item

        metric = {}
        for topk in self.topk_list:
            metric[topk] = self.topk_metric[topk](preds_scores, metric_target,
                                                  query_indexes)
        return metric

    def test_epoch_end(self, outputs):
        metric = {}
        for topk in self.topk_list:
            metric[topk] = {}
            for metric_name in outputs[0][topk].keys():
                scores = [m[topk][metric_name] for m in outputs]
                metric[topk][metric_name] = torch.stack(scores).mean()
                if METRIC_ABBR[metric_name] in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(
                    f"test_{METRIC_ABBR[metric_name]}@{topk}",
                    metric[topk][metric_name],
                    on_epoch=True,
                    prog_bar=log_on_progress_bar,
                    logger=True,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
