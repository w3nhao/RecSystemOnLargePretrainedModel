from dataclasses import dataclass
import dataclasses
from typing import List


@dataclass
class SeqRecConfig:
    item_token_num: int
    lr: float = 0.001
    sasrec_seq_len: int = 20
    sasrec_n_layers: int = 2
    sasrec_n_heads: int = 2
    sasrec_hidden_size: int = 64
    sasrec_inner_size: int = 256
    sasrec_hidden_dropout: float = 0.5
    sasrec_attention_dropout: float = 0.5
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    topk_list: int = dataclasses.field(default_factory=list)

@dataclass
class TextSeqRecConfig(SeqRecConfig):
    plm_name: str = 'facebook/opt-125m'
    plm_n_unfreeze_layers: int = 0
    plm_lr: float = 1e-5
    plm_lr_layer_decay: float = 0.8
    projection_n_layers: int = 5
    projection_inner_sizes: int = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert len(self.projection_inner_sizes) == self.projection_n_layers - 1

        if self.plm_n_unfreeze_layers == 0:
            del self.plm_lr
            del self.plm_lr_layer_decay

@dataclass
class OPTSeqRecConfig(TextSeqRecConfig):
    pooling_method: str = 'mean'

    def __post_init__(self):
        super().__post_init__()

        if self.pooling_method not in ['mean', 'last']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

@dataclass
class BERTSeqRecConfig(TextSeqRecConfig):
    pooling_method: str = 'cls'

    def __post_init__(self):
        super().__post_init__()

        if self.pooling_method not in ['cls', 'mean', 'pooler']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

@dataclass
class OPTPromptSeqRecConfig(OPTSeqRecConfig):
    pooling_method: str = 'mean_last'
    prompt_projection: str = 'nonlinear'
    prompt_hidden_size: int = 128
    pre_seq_len: int = 20
    post_seq_len: int = 10
    last_query_len: int = 1

    def __post_init__(self):
        super(OPTSeqRecConfig, self).__post_init__()

        if self.pooling_method not in ['mean_last', 'last', 'mean']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

        if self.prompt_projection not in ['nonlinear', 'linear']:
            raise ValueError(
                f"prompt_projection {self.prompt_projection} is not supported."
            )

        assert self.pre_seq_len >= 0
        assert self.post_seq_len >= 0
        assert self.last_query_len >= 0
        assert self.pre_seq_len + self.post_seq_len + self.last_query_len > 0

        if self.pooling_method in ['mean_last', 'last']:
            assert self.last_query_len > 0

@dataclass
class BERTPromptSeqRecConfig(BERTSeqRecConfig):
    pooling_method: str = 'cls'
    prompt_projection: str = 'nonlinear'
    prompt_hidden_size: int = 128
    pre_seq_len: int = 20

    def __post_init__(self):
        super(BERTSeqRecConfig, self).__post_init__()

        if self.pooling_method not in ['cls', 'mean', 'pooler']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

        if self.prompt_projection not in ['nonlinear', 'linear']:
            raise ValueError(
                f"prompt_projection {self.prompt_projection} is not supported."
            )

        assert self.pre_seq_len >= 0
