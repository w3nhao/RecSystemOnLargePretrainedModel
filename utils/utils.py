from models.recommender import (
    IDSeqRec,
    OPTSeqRec, 
    OPTPromptSeqRec, 
    BERTSeqRec, 
    BERTPromptSeqRec, 
)
from models.configs import (
    SeqRecConfig, 
    OPTPromptSeqRecConfig, 
    BERTPromptSeqRecConfig, 
    OPTSeqRecConfig, 
    BERTSeqRecConfig
)
from datamodules.configs import SeqRecDataModuleConfig

def add_program_args(parent_parser):
    parser = parent_parser.add_argument_group("program")
    parser.add_argument("--early_stop_patience",
                        type=int,
                        default=10,
                        help="early stop patience")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu")
    return parent_parser


def get_model(args):
    input_type = args.input_type
    plm_name = args.plm_name
    use_prompt = args.use_prompt
    if input_type == 'id':
        model = IDSeqRec
    elif input_type == 'text':
        if plm_name.startswith('facebook/opt'):
            if use_prompt:
                model = OPTPromptSeqRec
            else:
                model = OPTSeqRec
        elif plm_name.startswith('bert'):
            if use_prompt:
                model = BERTPromptSeqRec
            else:
                model = BERTSeqRec
    return model


def add_model_specific_args(args, parent_parser):
    input_type = args.input_type
    plm_name = args.plm_name
    use_prompt = args.use_prompt
    if input_type == 'id':
        parser = parent_parser.add_argument_group("IDSeqRec")
    else:
        if plm_name.startswith('facebook/opt'):
            if use_prompt:
                parser = parent_parser.add_argument_group("OPTPromptSeqRec")
                parser.add_argument("--post_seq_len", type=int, default=10)
                parser.add_argument("--last_query_len", type=int, default=1)
            else:
                parser = parent_parser.add_argument_group("OPTSeqRec")
        elif plm_name.startswith('bert'):
            if use_prompt:
                parser = parent_parser.add_argument_group("BERTPromptSeqRec")
            else:
                parser = parent_parser.add_argument_group("BERTSeqRec")

        # shared parameters of TextSeqRec
        parser.add_argument("--pooling_method", type=str, default="mean")
        parser.add_argument("--plm_n_unfreeze_layers", type=int, default=0)
        parser.add_argument("--projection_n_layers", type=int, default=5)
        parser.add_argument("--projection_inner_sizes",
                            type=int,
                            nargs="*",
                            default=[3136, 784, 3136, 784])
        
        # shared parameters of fine-tuneing PLM
        parser.add_argument("--plm_lr", type=float, default=1e-5)
        parser.add_argument("--plm_lr_layer_decay", type=float, default=0.8)

        # shared parameters of DeepPrompt
        if use_prompt:
            parser.add_argument("--pre_seq_len", type=int, default=20)
            parser.add_argument("--prompt_hidden_size", type=int, default=128)
            parser.add_argument("--prompt_projeciton",
                                type=str,
                                default="nonlinear")

    # shared parameters of SeqRec
    parser.add_argument("--lr", type=float, default=0.001)
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


def build_model_config(args, item_token_num):
    input_type = args.input_type
    plm_name = args.plm_name
    use_prompt = args.use_prompt

    if input_type == 'id':
        config = SeqRecConfig(item_token_num=item_token_num)
    elif input_type == 'text':
        if plm_name.startswith('facebook/opt'):
            if use_prompt:
                config = OPTPromptSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers = args.plm_n_unfreeze_layers,
                    plm_lr = args.plm_lr,
                    plm_lr_layer_decay = args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                    prompt_projection=args.prompt_projeciton,
                    prompt_hidden_size = args.prompt_hidden_size,
                    pre_seq_len=args.pre_seq_len,
                    post_seq_len=args.post_seq_len,
                    last_query_len=args.last_query_len,
                )
            else:
                config = OPTSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers = args.plm_n_unfreeze_layers,
                    plm_lr = args.plm_lr,
                    plm_lr_layer_decay = args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                )
        elif plm_name.startswith('bert'):
            if use_prompt:
                config = BERTPromptSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers = args.plm_n_unfreeze_layers,
                    plm_lr = args.plm_lr,
                    plm_lr_layer_decay = args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                    prompt_projection=args.prompt_projeciton,
                    prompt_hidden_size = args.prompt_hidden_size,
                    pre_seq_len=args.pre_seq_len,
                )
            else:
                config = BERTSeqRecConfig(
                    item_token_num=item_token_num,
                    plm_n_unfreeze_layers = args.plm_n_unfreeze_layers,
                    plm_lr = args.plm_lr,
                    plm_lr_layer_decay = args.plm_lr_layer_decay,
                    projection_n_layers=args.projection_n_layers,
                    projection_inner_sizes=args.projection_inner_sizes,
                    pooling_method=args.pooling_method,
                )

        config.plm_name = plm_name

    # shared parameters of SeqRec
    config.lr = args.lr
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


def add_datamodule_specific_args(parent_parser):
    """Add datamodule specific arguments to the parser."""

    def int_or_none(x):
        return None if x == "None" else int(x)

    parser = parent_parser.add_argument_group("SeqRecDataModule")
    parser.add_argument("--dataset", type=str, default="MIND_small")
    parser.add_argument("--min_item_seq_len", type=int, default=5)
    parser.add_argument("--max_item_seq_len", type=int_or_none, default=None)
    parser.add_argument("--sasrec_seq_len", type=int, default=20)
    parser.add_argument("--tokenized_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=False)
    return parent_parser


def build_datamodule_config(args):
    """Build configs from arguments."""
    config = SeqRecDataModuleConfig(
        dataset=args.dataset,
        plm_name=args.plm_name,
        min_item_seq_len=args.min_item_seq_len,
        max_item_seq_len=args.max_item_seq_len,
        sasrec_seq_len=args.sasrec_seq_len,
        tokenized_len=args.tokenized_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return config
