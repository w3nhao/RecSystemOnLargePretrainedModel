import argparse
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.recommender import SeqRecommender
from datamodules.datamodule import SeqDataModule, PRETRAIN_MODEL_ABBR
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
import torch.multiprocessing
from transformers import logging
from utils.pylogger import get_pylogger

torch.multiprocessing.set_sharing_strategy('file_system')

logging.set_verbosity_error()

log = get_pylogger(__name__)

# import wandb
# wandb.login
# wandb.init(project="my-opt-sas-new", entity="alanwake")

# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)

args = {
    "lr": 1e-3,
    "epochs": 200,
    "early_stop_patience": 10,
    "devices": [7],
    "batch_size": 128,
    "input_type": "id",
    "dataset": "MIND_small",
    "dim": 256,
    "num_blocks": 2,
    "num_heads": 2,
    "dropout": 0.5,
    "unfreeze": 0,
    "pretrained_model": "facebook/opt-125m",
    "sasrec_seq_len": 20,
    "tokenized_len": 30,
    "layer_norm_eps": 1e-6,
    "min_item_seq_len": 5,
    "max_item_seq_len": None,
    "no_grad": "no", 
    "use_mlp_connect": "no",
    "mlp_layers_num": 4,
    "mlp_inner_size": [3136, 784, 64],
    "num_workers": 0,
}

argparser = argparse.ArgumentParser()

argparser.add_argument("--devices",
                       type=int,
                       nargs="+",
                       default=args["devices"],
                       help="devices to use as list of int like 0 1 2"
                        " for cuda:0 cuda:1 cuda:2")

argparser.add_argument("--lr",
                       type=float,
                       default=args["lr"],
                       help="learning rate")

argparser.add_argument("--epochs",
                       type=int,
                       default=args["epochs"],
                       help="number of epochs")

argparser.add_argument("--batch_size",
                       type=int,
                       default=args["batch_size"],
                       help="batch size")

argparser.add_argument("--num_workers",
                       type=int,
                       default=args["num_workers"],
                       help="num_workers for dataloader")

argparser.add_argument("--early_stop_patience",
                       type=int,
                       default=args["early_stop_patience"],
                       help="early stop patience")

argparser.add_argument(
    "--dropout",
    type=float,
    default=args["dropout"],
    help="dropout prob of all dropout layers across the model")

argparser.add_argument(
    "--input_type",
    type=str,
    default=args["input_type"],
    help="input type of the model, only support 'id' and 'test'")

argparser.add_argument(
    "--tokenized_len",
    type=int,
    default=args["tokenized_len"],
    help="when input is text, each sentence will be tokenized to this length")

argparser.add_argument(
    "--dataset",
    type=str,
    default=args["dataset"],
    help="dataset name, only support 'MIND_large' and 'MIND_small'")

argparser.add_argument(
    "--min_item_seq_len",
    type=int,
    default=args["min_item_seq_len"],
    help=
    "minimum behaviors(interactions) sequence length of each user for filtering"
)

argparser.add_argument(
    "--max_item_seq_len",
    type=lambda x: None if x == "None" else int(x),
    default=args["max_item_seq_len"],
    help=
    "maximum behaviors(interactions) sequence length of each user for filtering"
)

argparser.add_argument(
    "--sasrec_seq_len",
    type=int,
    default=args["sasrec_seq_len"],
    help="behaviors(interactions) of each user will be cut to this length")

argparser.add_argument(
    "--dim",
    type=int,
    default=args["dim"],
    help="dim of both the embedding and sasrec hidden layer when input is id, "
    "when input is text this is the hidden size of sasrec")

argparser.add_argument("--num_blocks",
                       type=int,
                       default=args["num_blocks"],
                       help="the encoder blocks number of sasrec")

argparser.add_argument("--num_heads",
                       type=int,
                       default=args["num_heads"],
                       help="the attention heads number of encoder in sasrec")

argparser.add_argument("--layer_norm_eps",
                       type=float,
                       default=args["layer_norm_eps"],
                       help="eps for layer norm layers across the model")

argparser.add_argument(
    "--unfreeze",
    type=int,
    default=args["unfreeze"],
    help="unfreeze layers from the last, e.g. 1,2 or 0 for not unfreeze")

argparser.add_argument(
    "--no_grad",
    type=str,
    default=args["no_grad"],
    help="whether tatolly freeze the model which could save memory")

argparser.add_argument(
    "--pretrained_model",
    type=str,
    default=args["pretrained_model"],
    help=
    "pretrained model specified by name or path, e.g. "
    "'bert-base-uncased' or 'bert-base-uncased.tar.gz'"
)

argparser.add_argument(
    "--use_mlp_connect",
    type=str,
    default=args["use_mlp_connect"],
    help=
    "whether use mlp connect when input is text to connect the pretrained model and sasrec"
)

argparser.add_argument("--mlp_layers_num",
                       type=int,
                       default=args["mlp_layers_num"],
                       help="mlp layers number when use_mlp_connect is True")

argparser.add_argument(
    "--mlp_inner_size",
    type=int,
    nargs="+",
    default=args["mlp_inner_size"],
    help="mlp inner size when use_mlp_connect is True, "
    "the first and last dim would be set automatically as the same as the pretrained model and sasrec, "
    "so the length of this list should be mlp_layers_num - 2, e.g. [784 * 4, 784, 64]"
)


args = argparser.parse_args()

args.no_grad = True if args.no_grad == "yes" else False
args.use_mlp_connect = True if args.use_mlp_connect == "yes" else False

dm = SeqDataModule(
    data_name=args.dataset,
    batch_size=args.batch_size,
    pretrained_model=args.pretrained_model,
    sasrec_seq_len=args.sasrec_seq_len,
    tokenized_len=args.tokenized_len,
    min_item_seq_len=args.min_item_seq_len,
    max_item_seq_len=args.max_item_seq_len,
    num_workers=args.num_workers,
)

num_items = dm.prepare_data()

model = SeqRecommender(
    lr=args.lr,
    item_token_num=num_items,
    sasrec_seq_len=args.sasrec_seq_len,
    num_unfreeze_layers=args.unfreeze,
    no_grad=args.no_grad,
    input_type=args.input_type,
    pretrained_model=args.pretrained_model,
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


if args.input_type == "id":
    model_name = "SASRecWithID"
    base_model_name = "EMB"
elif args.input_type == "text":
    model_name = "SASRecWithText"
    if args.pretrained_model.startswith("facebook"):
        base_model_name = PRETRAIN_MODEL_ABBR[args.pretrained_model]
    elif args.pretrained_model.startswith("google"):
        base_model_name = "BERT"
    else:
        raise ValueError("Unknown backbone name")

exec_time = datetime.now().strftime("%y%m%d%H%M%S")
devices_name = "cuda:" + "".join([str(i) for i in args.devices])
version_name = f"{base_model_name}_{exec_time}"
checkpoint_callback = ModelCheckpoint(
    dirpath=f"logs/{devices_name}/{args.dataset}/{model_name}/{version_name}",
    save_top_k=1,
    monitor="val_HR@10",
    mode="max",
    filename="-{epoch:02d}-{val_HR@10:.2f}",
)

early_stop_callback = EarlyStopping(monitor="val_HR@10",
                                    mode="max",
                                    patience=args.early_stop_patience)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"logs/{devices_name}/{args.dataset}",
                                         name=model_name,
                                         version=version_name)

csv_logger = pl_loggers.CSVLogger(save_dir=f"logs/{devices_name}/{args.dataset}",
                                  name=model_name,
                                  version=version_name,)

trainer = Trainer(
    logger=[tb_logger, csv_logger],
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=args.devices,
    deterministic=True,
    callbacks=[checkpoint_callback, early_stop_callback],
    # fast_dev_run=100,
)

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm, ckpt_path="best")
