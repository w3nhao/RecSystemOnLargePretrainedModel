class SeqRecConfig:
    def __init__(self, item_token_num: int, **kwargs):
        self.item_token_num = item_token_num
        self.lr = kwargs.pop("lr", 1e-3)
        self.weight_decay = kwargs.pop("weight_decay", 0.1)
        self.sasrec_seq_len = kwargs.pop("sasrec_seq_len", 20)
        self.sasrec_n_layers = kwargs.pop("sasrec_n_layers", 2)
        self.sasrec_n_heads = kwargs.pop("sasrec_n_heads", 2)
        self.sasrec_hidden_size = kwargs.pop("sasrec_hidden_size", 64)
        self.sasrec_inner_size = kwargs.pop("sasrec_inner_size", 256)
        self.sasrec_hidden_dropout = kwargs.pop("sasrec_hidden_dropout", 0.5)
        self.sasrec_attention_dropout = kwargs.pop("sasrec_attention_dropout", 0.5)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        self.topk_list = kwargs.pop("topk_list", [5, 10, 20])

        if kwargs:
            raise ValueError(f'Unrecognized arguments: {kwargs}')
        

class TextSeqRecConfig(SeqRecConfig):
    def __init__(self, item_token_num: int, **kwargs):
        self.plm_name = kwargs.pop("plm_name", 'facebook/opt-125m')
        self.plm_n_unfreeze_layers = kwargs.pop("plm_n_unfreeze_layers", 0)
        
        plm_lr = kwargs.pop("plm_lr", 1e-5)
        plm_lr_layer_decay = kwargs.pop("plm_lr_layer_decay", 0.8)
        if self.plm_n_unfreeze_layers != 0:
            self.plm_lr = plm_lr
            self.plm_lr_layer_decay = plm_lr_layer_decay
            
        self.projection_n_layers = kwargs.pop("projection_n_layers", 5)
        self.projection_inner_sizes = kwargs.pop("projection_inner_sizes", [3136, 784, 3136])
        assert len(self.projection_inner_sizes) == self.projection_n_layers - 1
        
        super().__init__(item_token_num, **kwargs)
        
        
class OPTSeqRecConfig(TextSeqRecConfig):
    def __init__(
        self, 
        item_token_num: int, 
        pooling_method: str = 'mean', 
        **kwargs
        ):
        self.pooling_method = pooling_method
        
        if self.pooling_method not in ['mean', 'last']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")
        
        super().__init__(item_token_num, **kwargs)


class BERTSeqRecConfig(TextSeqRecConfig):
    def __init__(
        self, 
        item_token_num: int, 
        pooling_method: str = 'cls', 
        **kwargs
        ):
        self.pooling_method = pooling_method
 
        if self.pooling_method not in ['cls', 'mean', 'pooler']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")
        
        super().__init__(item_token_num, **kwargs)


class OPTPromptSeqRecConfig(TextSeqRecConfig):
    def __init__(
        self, 
        item_token_num: int, 
        pooling_method: str = 'mean_last',
        prompt_projection: str = 'nonlinear',
        prompt_hidden_size: int = 128,
        pre_seq_len: int = 20,
        post_seq_len: int = 10,
        last_query_len: int = 1,
        **kwargs
        ):
        self.pooling_method = pooling_method
        self.prompt_projection = prompt_projection
        self.prompt_hidden_size = prompt_hidden_size
        self.pre_seq_len = pre_seq_len
        self.post_seq_len = post_seq_len
        self.last_query_len = last_query_len

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
            
        super().__init__(item_token_num, **kwargs)


class BERTPromptSeqRecConfig(TextSeqRecConfig):
    def __init__(
        self, 
        item_token_num: int, 
        pooling_method: str = 'cls',
        prompt_hidden_size: int = 128,
        prompt_projection: str = 'nonlinear',
        pre_seq_len: int = 20,
        **kwargs
        ):
        self.pooling_method = pooling_method
        self.prompt_hidden_size = prompt_hidden_size
        self.pre_seq_len = pre_seq_len
        self.prompt_projection = prompt_projection


        if self.pooling_method not in ['cls', 'mean', 'pooler']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

        if self.prompt_projection not in ['nonlinear', 'linear']:
            raise ValueError(
                f"prompt_projection {self.prompt_projection} is not supported."
            )

        assert self.pre_seq_len >= 0

        super().__init__(item_token_num, **kwargs)
