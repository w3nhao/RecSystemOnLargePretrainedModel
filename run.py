import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from transformers import logging

from utils import (
    add_program_args, 
    get_program_details, 
    read_distributed_strategy,
    get_model,
    get_datamodule,
)

from utils.pylogger import get_pylogger

torch.multiprocessing.set_sharing_strategy('file_system')
logging.set_verbosity_error()
log = get_pylogger(__name__)


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
    
    # set program args
    temp_args, _ = argparser.parse_known_args()
    argparser = add_program_args(temp_args, argparser)

    # set model and dataset args
    temp_args, _ = argparser.parse_known_args()
    rec_model = get_model(args=temp_args)
    argparser = rec_model.add_model_specific_args(parent_parser=argparser)
    datamodule = get_datamodule(args=temp_args)
    argparser = datamodule.add_datamodule_specific_args(parent_parser=argparser)

    # parse args
    args, _ = argparser.parse_known_args()

    # set up datamodule
    datamodule_config = datamodule.build_datamodule_config(args=args)
    dm = datamodule(datamodule_config)

    # prepare data
    num_items = dm.prepare_data()
    
    # build model
    model_config = rec_model.build_model_config(args=args,
                                                item_token_num=num_items)
    model = rec_model(model_config)

    # set up trainer
    model_name, version_name, devices_name = get_program_details(args)
    ckpt_path = f"logs/{devices_name}/{args.dataset}/{model_name}/{version_name}"
    log_save_dir = f"logs/{devices_name}/{args.dataset}"

    # set up callbacks
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

    # set up logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_save_dir,
                                             name=model_name,
                                             version=version_name)
    csv_logger = pl_loggers.CSVLogger(save_dir=log_save_dir,
                                      name=model_name,
                                      version=version_name)

    # set up profiler
    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

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
        # test on a single accelerator
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
