from abc import abstractmethod, ABC
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from utils.metrics import MRR, NDCG, HR
from utils.pylogger import get_pylogger
from models.sasrec import SASRec
from models.configs import SeqRecConfig, TextSeqRecConfig
from models.utils import gather_indexes

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
    def _feature_extract(self, input):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    @abstractmethod
    def _val_test_step(self, batch, batch_idx, stage):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, "test")
    
    def _val_test_epoch_end(self, outputs, stage):
        topk_list = self.hparams.config.topk_list
        for topk in topk_list:
            for metric_name in METRIC_LIST:
                score = self.topk_metric[f"{metric_name}@{topk}"].compute()
                if metric_name in ["HR", "NDCG"] and topk == 10:
                    log_on_progress_bar = True
                else:
                    log_on_progress_bar = False
                self.log(f"{stage}_{metric_name}@{topk}",
                         score,
                         on_epoch=True,
                         prog_bar=log_on_progress_bar,
                         logger=True,
                         sync_dist=True)
    
    def validation_epoch_end(self, outputs):
        self._val_test_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._val_test_epoch_end(outputs, "test")
    
    def configure_optimizers(self):
        raise NotImplementedError

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
        parser.add_argument("--sasrec_attention_dropout",
                            type=float,
                            default=0.5)
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


class TextSeqRec(SeqRec, ABC):

    def __init__(self, config: TextSeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

    def _set_feature_extractor(self, config):
        plm_name = config.plm_name
        last_n_unfreeze_layers = config.plm_last_n_unfreeze
        self._set_plm_model(plm_name)
        self._freeze_plm_layers(last_n_unfreeze_layers)

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
    def _freeze_plm_layers(self, last_n_unfreeze_layers):
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
                            default=[3072, 768, 3072, 768])
        return parent_parser

    @classmethod
    def build_model_config(cls, args, config):
        config = super(TextSeqRec, cls).build_model_config(args, config)
        config.plm_name = args.plm_name
        return config