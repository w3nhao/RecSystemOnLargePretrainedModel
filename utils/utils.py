from datetime import datetime
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from datamodules import (
    SeqDataModule,
    PreInferSeqDataModule,
    AllFreezePreInferSeqDataModule,
    PointWiseDataModule,
)
from models import (
    IDSeqRec,
    OPTSeqRec,
    OPTPromptSeqRec,
    BERTSeqRec,
    BERTPromptSeqRec,
    PreInferOPTSeqRec,
)
from utils.cli_parse import parse_boolean
from models.utils import PRETRAIN_MODEL_ABBR

def get_model(args):
    if args.architecture == "dssm":
        if args.input_type == 'id':
            model = None
        elif args.input_type == 'text':
            if args.plm_name.startswith('facebook/opt'):
                if args.pre_inference:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            elif args.plm_name.startswith('bert'):
                if args.pre_inference:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
    elif args.architecture == "sasrec":
        if args.input_type == 'id':
            model = IDSeqRec
        elif args.input_type == 'text':
            if args.plm_name.startswith('facebook/opt'):
                if args.pre_inference:
                    if args.use_prompt:
                        raise NotImplementedError
                    else:
                        model = PreInferOPTSeqRec
                else:
                    if args.use_prompt:
                        model = OPTPromptSeqRec
                    else:
                        model = OPTSeqRec
            elif args.plm_name.startswith('bert'):
                if args.pre_inference:
                    if args.use_prompt:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    if args.use_prompt:
                        model = BERTPromptSeqRec
                    else:
                        model = BERTSeqRec
    else:
        raise NotImplementedError
    return model


def get_datamodule(args):
    # add new datamodule if needed
    if args.architecture == "dssm":
        if args.input_type == 'id':
            data_module = PointWiseDataModule
        elif args.input_type == 'text':
            if args.pre_inference:
                if args.plm_last_n_unfreeze == 0:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                data_module = PointWiseDataModule
    elif args.architecture == "sasrec":
        if args.input_type == 'id':
            data_module = SeqDataModule
        elif args.input_type == 'text':
            if args.pre_inference:
                if args.plm_last_n_unfreeze == 0:
                    data_module = AllFreezePreInferSeqDataModule
                else:
                    data_module = PreInferSeqDataModule
            else:
                data_module = SeqDataModule
    else:
        raise NotImplementedError
    return data_module


def get_program_details(args):
    if args.input_type == "id":
        model_name = "SASRecWithID"
        base_model_name = "EMB"
    elif args.input_type == "text":
        if args.plm_name in PRETRAIN_MODEL_ABBR:
            model_name = "SASRecWithText"
            base_model_name = PRETRAIN_MODEL_ABBR[args.plm_name]
        else:
            raise ValueError("Unknown backbone name")

    precision = args.precision
    exec_time = datetime.now().strftime("%y%m%d%H%M%S")
    devices_name = "cuda:" + "".join([str(i) for i in args.devices])
    version_name = f"{exec_time}_{base_model_name}_PRC@{precision}"

    return model_name, version_name, devices_name


def add_program_args(args, parent_parser):
    parser = parent_parser.add_argument_group("program")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--early_stop_patience",
                        type=int,
                        default=10,
                        help="early stop patience")
    parser.add_argument(
        "--strategy",
        type=str,
        default="none",
        help="specify deepspeed stage, support stage_2, stage_2_offload"
        "stage_3, stage_3_offload, none")

    if args.input_type == 'text':
        parser.add_argument("--plm_name",
                            type=str,
                            default="facebook/opt-125m")
        parser.add_argument("--use_prompt",
                            type=parse_boolean,
                            default=False,
                            help="whether to use prompt")
        parser.add_argument("--pre_inference",
                            type=parse_boolean,
                            default=False,
                            help="whether to pre-inference")

    return parent_parser

def read_distributed_strategy(args):
    if args.strategy == "none":
        strategy = "ddp" if len(args.devices) > 1 else None
    elif args.strategy == "ddp_find_unused_parameters_false":
        strategy = "ddp_find_unused_parameters_false"
    elif args.strategy == "stage_2":
        strategy = "deepspeed_stage_2"
    elif args.strategy == "stage_2_offload":
        strategy = "deepspeed_stage_2_offload"
    elif args.strategy == "stage_3":
        strategy = "deepspeed_stage_3"
    elif args.strategy == "stage_3_offload":
        strategy = "deepspeed_stage_3_offload"
    elif args.strategy == "fsdp_offload":
        strategy = DDPFullyShardedNativeStrategy(
            cpu_offload=CPUOffload(offload_params=True)
            )
    else:
        raise ValueError("Unsupport strategy: {}".format(args.strategy))
    return strategy
