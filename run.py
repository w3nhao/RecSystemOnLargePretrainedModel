import argparse
import torch
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from transformers import logging

from datamodules.datamodule import SeqDataModule, PRETRAIN_MODEL_ABBR
from utils.pylogger import get_pylogger
from utils.utils import (
    add_program_args,
    add_model_specific_args, 
    get_model, 
    build_model_config, 
    add_datamodule_specific_args, 
    build_datamodule_config
)


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
    argparser.add_argument("--plm_name",
                           type=str,
                           default="facebook/opt-125m",
                           help="pretrained language model name")
    argparser.add_argument("--use_prompt",
                           type=parse_boolean,
                           default=False,
                           help="whether to use prompt")

    temp_args, _ = argparser.parse_known_args()
    rec_model = get_model(args=temp_args)

    argparser = add_model_specific_args(args=temp_args, parent_parser=argparser)
    argparser = add_datamodule_specific_args(argparser)
    argparser = add_program_args(argparser)
    args, _ = argparser.parse_known_args()

    # set up datamodule
    datamodule_config = build_datamodule_config(args=args)
    dm = SeqDataModule(datamodule_config)
    num_items = dm.prepare_data()

    # set up model
    model_config = build_model_config(args=args, item_token_num=num_items)
    model = rec_model(model_config)

    # set up trainer
    model_name, version_name, devices_name = get_details(args)
    ckpt_path = f"logs/{devices_name}/{args.dataset}/{model_name}/{version_name}"
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

    log_save_dir = f"logs/{devices_name}/{args.dataset}"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_save_dir,
                                             name=model_name,
                                             version=version_name)
    csv_logger = pl_loggers.CSVLogger(save_dir=log_save_dir,
                                      name=model_name,
                                      version=version_name)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[tb_logger, csv_logger],
        deterministic=True,
        strategy="ddp" if len(args.devices) > 1 else None,
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
