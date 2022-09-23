import os
import argparse
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.recommender import SeqRecommender
from datamodules.datamodule import SeqDataModule, PRETRAIN_MODEL_ABBR


from transformers import logging

logging.set_verbosity_error()

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# import wandb
# wandb.login
# wandb.init(project="my-opt-sas-new", entity="alanwake")

# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)

for path in ["logs", "models"]:
    if not os.path.exists(path):
        os.mkdir(path)

args = {
    "lr": 0.00001,
    "finetune_lr": "None",
    "epochs": 200,
    "device": ["cuda:6"],
    "batch_size": 16,
    "input_type": "id",
    "dataset": "MIND_large",
    "dim": 64,
    "num_blocks": 2,
    "num_heads": 2,
    "dropout": 0.1,
    # unfreeze layers from the last, e.g. 1,2 or None for not unfreeze
    "unfreeze": 0,
    "opt": "facebook/opt-125m",
    "sasrec_seq_len": 20,
    "tokenized_len": 30,
    "layer_norm_eps": 1e-12,
    "min_item_seq_len": 5,
    "max_item_seq_len": None,
    "no_grad": False,  # tatolly freeze and save memory
    "use_mlp_connect": False,
    "mlp_layers_num": 4,
    "mlp_inner_size": [784 * 4, 64]
}

argparser = argparse.ArgumentParser()

argparser.add_argument("--lr", type=float, default=args["lr"])
argparser.add_argument("--finetune_lr", type=str, default=args["finetune_lr"])
argparser.add_argument("--epochs", type=int, default=args["epochs"])
argparser.add_argument("--device", type=str, default=args["device"])
argparser.add_argument("--batch_size", type=int, default=args["batch_size"])
argparser.add_argument("--input_type", type=str, default=args["input_type"])
argparser.add_argument("--dataset", type=str, default=args["dataset"])
argparser.add_argument("--dim", type=int, default=args["dim"])
argparser.add_argument("--num_blocks", type=int, default=args["num_blocks"])
argparser.add_argument("--num_heads", type=int, default=args["num_heads"])
argparser.add_argument("--dropout", type=float, default=args["dropout"])
argparser.add_argument("--unfreeze", type=int, default=args["unfreeze"])
argparser.add_argument("--opt", type=str, default=args["opt"])
argparser.add_argument("--no_grad", type=bool, default=args["no_grad"])
argparser.add_argument("--use_mlp_connect",
                       type=bool,
                       default=args["use_mlp_connect"])
argparser.add_argument("--mlp_layers_num",
                       type=int,
                       default=args["mlp_layers_num"])
argparser.add_argument("--mlp_inner_size",
                       type=list,
                       default=args["mlp_inner_size"])
argparser.add_argument("--tokenized_len",
                       type=int,
                       default=args["tokenized_len"])
argparser.add_argument("--sasrec_seq_len",
                       type=int,
                       default=args["sasrec_seq_len"])
argparser.add_argument("--layer_norm_eps",
                       type=float,
                       default=args["layer_norm_eps"])
argparser.add_argument("--min_item_seq_len",
                       type=int,
                       default=args["min_item_seq_len"])
argparser.add_argument("--max_item_seq_len",
                       type=int,
                       default=args["max_item_seq_len"])

args = argparser.parse_args()

dm = SeqDataModule(
    data_name=args.dataset,
    batch_size=args.batch_size,
    pretrained_model=args.opt,
    sasrec_seq_len=args.sasrec_seq_len,
    tokenized_len=args.tokenized_len,
    min_item_seq_len=args.min_item_seq_len,
    max_item_seq_len=args.max_item_seq_len,
)

num_items = dm.prepare_data()

model = SeqRecommender(
    lr=args.lr,
    item_token_num=num_items,
    sasrec_seq_len=args.sasrec_seq_len,
    num_unfreeze_layers=args.unfreeze,
    no_grad=args.no_grad,
    input_type=args.input_type,
    pretrained_model=args.opt,
    n_layers=args.num_blocks,
    n_heads=args.num_heads,
    hidden_size=args.dim,
    inner_size=args.dim * 4,
    hidden_dropout=args.dropout,
    attention_dropout=args.dropout,
    layer_norm_eps=args.layer_norm_eps,
    initializer_range=0.02,
    use_mlp_connect=args.use_mlp_connect,
    mlp_layers_num=args.mlp_layers_num,
    mlp_inner_size=args.mlp_inner_size,
)

devices = [int(d[-1]) for d in args.device]

if args.input_type == "id":
    model_name = "SASRecWithID"
    backbone_name = "EMB"
elif args.input_type == "Text":
    model_name = "SASRecWithText"
    if args.opt.startswith("facebook"):
        backbone_name = PRETRAIN_MODEL_ABBR[args.opt]
    elif args.opt.startswith("google"):
        backbone_name = "BERT"
    else:
        raise ValueError("Unknown backbone name")

exec_time = datetime.now().strftime('%m%d%H%M%S')
version_name = f"{backbone_name}_{exec_time}"

checkpoint_callback = ModelCheckpoint(
    dirpath=f"logs/{model_name}/{version_name}",
    save_top_k=1,
    monitor="val_HR@10",
    mode="max",
    filename="-{epoch:02d}-{val_HR@10:.2f}")

early_stop_callback = EarlyStopping(monitor="val_HR@10",
                                    mode="max",
                                    patience=5)

tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/",
                                         name=model_name,
                                         version=version_name)

csv_logger = pl_loggers.CSVLogger(save_dir="logs/",
                                  name=model_name,
                                  version=version_name)

trainer = Trainer(
    logger=[tb_logger, csv_logger],
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=devices,
    deterministic=True,
    callbacks=[checkpoint_callback, early_stop_callback],
    # fast_dev_run=2,
)

trainer.fit(model, datamodule=dm)


