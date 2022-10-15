from datetime import datetime
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from datamodules.utils import PRETRAIN_MODEL_ABBR


def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


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
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu")
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


def add_pre_inference_args(args, parent_parser):
    if args.input_type == "text" and args.pre_inference:
        parser = parent_parser.add_argument_group("pre_inference")
        parser.add_argument("--pre_inference_batch_size", type=int, default=1)
        parser.add_argument("--pre_inference_devices",
                            type=int,
                            nargs="+",
                            default=[0])
        parser.add_argument("--pre_inference_precision", type=int, default=32)
    return parent_parser


def read_distributed_strategy(args):
    if args.strategy == "none":
        strategy = "ddp" if len(args.devices) > 1 else None
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
