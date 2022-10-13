import argparse
import torch
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers import logging

from datamodules.datamodule import (
    SeqDataModule, 
    PRETRAIN_MODEL_ABBR
)
from models.recommender import (
    IDSeqRec,
    OPTSeqRec, 
    OPTPromptSeqRec, 
    BERTSeqRec, 
    BERTPromptSeqRec, 
)
from utils.pylogger import get_pylogger

torch.multiprocessing.set_sharing_strategy('file_system')

logging.set_verbosity_error()

log = get_pylogger(__name__)

# profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")


def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


def get_details(args):
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
    parser.add_argument("--early_stop_patience",
                        type=int,
                        default=10,
                        help="early stop patience")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument(
        "--strategy",
        type=str,
        default="none",
        help="specify deepspeed stage, support stage_2, stage_2_offload"
        "stage_3, stage_3_offload, none")
    
    if args.input_type == 'text':
        parser.add_argument("--plm_name", type=str, default="facebook/opt-125m")
        parser.add_argument("--pre_inference",
                           type=parse_boolean,
                           default=False,
                           help="whether to do pre-inference")
        parser.add_argument("--use_prompt",
                            type=parse_boolean,
                            default=False,
                            help="whether to use prompt")
        
    return parent_parser


def get_model(args):
    input_type = args.input_type
    if input_type == 'id':
        model = IDSeqRec
    elif input_type == 'text':
        plm_name = args.plm_name
        use_prompt = args.use_prompt
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
        strategy = DDPFullyShardedNativeStrategy(cpu_offload=CPUOffload(
            offload_params=True))
    else:
        raise ValueError("Unsupport strategy: {}".format(args.strategy))
    return strategy


if __name__ == "__main__":

    seed_everything(42, workers=True)

    # ------------------------
    # SETTINGS
    # ------------------------

    # set up CLI args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_type",
                           type=str,
                           default="id",
                           help="input type of the model, "
                           "only support 'id' and 'test'")

    # get input type and strategy
    temp_args, _ = argparser.parse_known_args()
    argparser = add_program_args(temp_args, argparser)
    
    # if input type is text, add plm args
    temp_args, _ = argparser.parse_known_args()
    rec_model = get_model(args=temp_args)
    argparser = rec_model.add_model_specific_args(parent_parser=argparser)
    
    # add datamodule args
    datamodule = SeqDataModule
    argparser = datamodule.add_datamodule_specific_args(argparser)
    args, _ = argparser.parse_known_args()

    # set up datamodule
    datamodule_config = datamodule.build_datamodule_config(args=args)
    dm = datamodule(datamodule_config)
    num_items = dm.prepare_data()

    # set up model
    model_config = rec_model.build_model_config(args=args, item_token_num=num_items)
    model = rec_model(model_config)

    # set up trainer
    model_name, version_name, devices_name = get_details(args)
    ckpt_path = f"logs/{devices_name}/{args.dataset}/{model_name}/{version_name}"
    log_save_dir = f"logs/{devices_name}/{args.dataset}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=1,
        monitor="val_HR@10",
        mode="max",
        filename="-{epoch:02d}-{val_HR@10:.2f}",
    )
    early_stop_callback = EarlyStopping(monitor="val_HR@10",
                                        mode="max",
                                        patience=args.early_stop_patience)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_save_dir,
                                             name=model_name,
                                             version=version_name)
    csv_logger = pl_loggers.CSVLogger(save_dir=log_save_dir,
                                      name=model_name,
                                      version=version_name)

    strategy = read_distributed_strategy(args)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[tb_logger, csv_logger],
        deterministic=True,
        strategy=strategy,
        # val_check_interval=0.25,
        # strategy="ddp_find_unused_parameters_false" if len(args.devices) > 1 else None,
    )

    # ------------------------
    # START TRAINING
    # ------------------------

    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)

    # ------------------------
    # START TESTING
    # ------------------------
    if len(args.devices) == 1:
        trainer.test(datamodule=dm, ckpt_path="best")
    elif len(args.devices) > 1:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        test_logger = pl_loggers.CSVLogger(
            save_dir=f"logs/{devices_name}/{args.dataset}",
            name=model_name,
            version=version_name + "_test")
        tester = Trainer(logger=test_logger,
                         accelerator="gpu",
                         devices=[args.devices[0]],
                         deterministic=True,
                         precision=32)
        tester.test(model, ckpt_path=ckpt_path, datamodule=dm)
