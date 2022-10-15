import sys
import os
# add the realpath of parent directory to the path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))
# delete the current directory from the path
sys.path.remove(current_dir)


import argparse
import torch
import argparse
import pytorch_lightning as pl
import pandas as pd
from transformers import OPTModel, BertModel
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from datamodules.utils import (
    str_fields2ndarray,
    TEXT_ID_SEQ_FIELD,
    ATTENTION_MASK_FIELD,
    PRETRAIN_MODEL_ABBR,
)

from transformers import logging
logging.set_verbosity_error()


class PLMDataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

    def __len__(self):
        return len(self.input_ids)


class PredictWriter(BasePredictionWriter):

    def __init__(self, output_dir, plm_name, n_freeze_layers,
                 n_unfreeze_layers, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.plm_abbr = PRETRAIN_MODEL_ABBR[plm_name]
        self.n_freeze_layers = n_freeze_layers
        self.n_unfreeze_layers = n_unfreeze_layers

    def write_on_epoch_end(self, trainer, pl_module, predictions,
                           batch_indices):
        rank_idx = pl_module.global_rank
        predictions = torch.cat(predictions[0], dim=0)
        batch_indices = [idx for idxs in batch_indices[0] for idx in idxs]
        batch_indices = torch.tensor(batch_indices)
        torch.save(
            predictions,
            os.path.join(
                self.output_dir, f"{self.plm_abbr}_"
                f"freeze@{self.n_freeze_layers}_"
                f"inferenced_embs_for_"
                f"unfreeze@{self.n_unfreeze_layers}_"
                f"{rank_idx}.pt"))
        torch.save(
            batch_indices,
            os.path.join(
                self.output_dir, f"{self.plm_abbr}_"
                f"freeze@{self.n_freeze_layers}_"
                f"inferenced_idxs_for_"
                f"unfreeze@{self.n_unfreeze_layers}_"
                f"{rank_idx}.pt"))


class FrozenPartialPLM(pl.LightningModule):

    def __init__(self, plm_name, plm_n_unfreeze_layers):
        super(FrozenPartialPLM, self).__init__()

        self.plm_name = plm_name
        self.plm_n_unfreeze_layers = plm_n_unfreeze_layers
        assert self.plm_n_unfreeze_layers >= 0

        # use all layers of the plm model except the last n unfreeze layers
        if self.plm_name.startswith("facebook"):
            self.plm = OPTModel.from_pretrained(plm_name)
            assert self.plm_n_unfreeze_layers < self.plm.config.num_hidden_layers
            if self.plm_n_unfreeze_layers > 0:
                self.plm.decoder.layers = \
                    self.plm.decoder.layers[:-self.plm_n_unfreeze_layers]
            else:
                self.plm.decoder.layers = self.plm.decoder.layers

        elif self.plm_name.startswith("bert"):
            assert self.plm_n_unfreeze_layers < self.plm.config.num_hidden_layers
            self.plm = BertModel.from_pretrained(plm_name)
            if self.plm_n_unfreeze_layers > 0:
                self.plm.encoder.layer = \
                    self.plm.encoder.layer[:-self.plm_n_unfreeze_layers]
            else:
                self.plm.encoder.layer = self.plm.encoder.layer

        self.n_freeze_layers = self.plm.config.num_hidden_layers - self.plm_n_unfreeze_layers

        for params in self.plm.parameters():
            params.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.plm(input_ids, attention_mask)
        return output

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        output = self(input_ids, attention_mask)
        return output.last_hidden_state


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--processed_dir", type=str)
    argparser.add_argument("--processed_items_file", type=str)
    argparser.add_argument("--plm_name", type=str, default="facebook/opt-125m")
    argparser.add_argument("--plm_n_unfreeze_layers", type=int, default=0)
    argparser.add_argument("--tokenized_len", type=int, default=30)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--num_workers", type=int, default=4)
    argparser.add_argument("--devices", type=int, nargs="+", default=[0])
    argparser.add_argument("--precision", type=int, default=32)
    args, _ = argparser.parse_known_args()

    tokenized_len = args.tokenized_len
    items_path = os.path.join(args.processed_dir, args.processed_items_file)
    items = pd.read_csv(
        items_path,
        sep="\t",
        header=0,
    )

    num_items = len(items)

    tokenized_ids, attention_mask = str_fields2ndarray(
        df=items,
        fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
        field_len=tokenized_len,
    )
    items = PLMDataset(tokenized_ids, attention_mask)
    items_loader = DataLoader(items,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    model = FrozenPartialPLM(args.plm_name, args.plm_n_unfreeze_layers)
    n_freeze_layers = model.n_freeze_layers

    inferencer = Trainer(
        accelerator='gpu',
        devices=args.devices,
        precision=args.precision,
        strategy='ddp' if len(args.devices) > 1 else None,
        callbacks=[
            PredictWriter(output_dir=args.processed_dir,
                          plm_name=args.plm_name,
                          n_freeze_layers=n_freeze_layers,
                          n_unfreeze_layers=args.plm_n_unfreeze_layers,
                          write_interval="epoch")
        ],
        logger=False,
        deterministic=True,
        enable_model_summary=True,
    )
    inferenced_data = inferencer.predict(model, items_loader)