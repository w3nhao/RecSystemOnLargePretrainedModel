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
import numpy as np
from transformers import BertModel, AutoConfig
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from utils.cli_parse import str_or_none, parse_plm_layer
from datamodules.utils import PRETRAIN_MODEL_ABBR
from models.partial_opt import PartialOPTModel

from transformers import logging
logging.set_verbosity_error()


class TokenizedDataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

    def __len__(self):
        return len(self.input_ids)
    

class EmbeddingsDataset(TokenizedDataset):

    def __init__(self, input_ids, attention_mask, input_embeds):
        super().__init__(input_ids, attention_mask)
        self.input_embeds = input_embeds
    
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.input_embeds[index]
    
    def __len__(self):
        return super().__len__()


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

    def __init__(self,
                 plm_name,
                 keep_embed_layer,
                 keep_inner_layers_range,
                 ):
        """ keep inner layers range is a tuple of both inclusive index (start, end) """
        super(FrozenPartialPLM, self).__init__()
        self.plm_name = plm_name
        if self.plm_name.startswith("facebook"):
            self.plm_config = AutoConfig.from_pretrained(self.plm_name)
            self.plm = PartialOPTModel.from_pretrained(
                plm_name,
                keep_embed_layer=keep_embed_layer,
                keep_decoders_range=keep_inner_layers_range)

        elif self.plm_name.startswith("bert"):
            self.plm = BertModel.from_pretrained(plm_name)
            if self.plm_last_n_unfreeze > 0:
                self.plm.encoder.layer = \
                    self.plm.encoder.layer[:-self.plm_last_n_unfreeze]
            else:
                self.plm.encoder.layer = self.plm.encoder.layer

        self.n_freeze_layers = self.plm.config.num_hidden_layers - self.plm_last_n_unfreeze

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
    argparser.add_argument("--plm_name", type=str, default="facebook/opt-125m")
    argparser.add_argument("--processed_dir", type=str, required=True)
    argparser.add_argument("--processed_items_file", type=str, required=True)
    argparser.add_argument("--input_item_embs_file", type=str_or_none, default=None)
    argparser.add_argument("--input_layer", type=parse_plm_layer, default="none")
    argparser.add_argument("--target_layer", type=parse_plm_layer, default="emb")
    argparser.add_argument("--tokenized_len", type=int, default=30)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--num_workers", type=int, default=4)
    argparser.add_argument("--devices", type=int, nargs="+", default=[0])
    argparser.add_argument("--precision", type=int, default=32)
    args, _ = argparser.parse_known_args()

    # get the plm config
    plm_config = AutoConfig.from_pretrained(args.plm_name)
    
    levels = ["none", "emb"] + [i for i in range(plm_config.num_hidden_layers)]
    assert args.input_layer in levels
    assert args.target_layer in levels
    
    input_level_idx = levels.index(args.input_layer)
    target_level_idx = levels.index(args.target_layer)
    assert target_level_idx < plm_config.num_hidden_layers
    assert input_level_idx < target_level_idx
    
    # load the processed input_ids and attention_mask
    items_path = os.path.join(args.processed_dir, args.processed_items_file)
    items = np.load(items_path, allow_pickle=True)
    input_ids = items[0]
    attention_mask = items[1]
    
    # set the dataset
    # load the input item embeddings if needed
    if input_level_idx > 0:
        assert args.input_item_embs_file is not None
        item_embs_path = os.path.join(args.processed_dir, args.input_item_embs_file)
        items_embs = torch.load(item_embs_path)
        items = EmbeddingsDataset(input_ids, attention_mask, items_embs)
    else:
        items = EmbeddingsDataset(input_ids, attention_mask)
    
    # set the dataloader
        items_loader = DataLoader(items,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        
    # set the model
    keep_embed_layer = True if input_level_idx == 0 else False
    if target_level_idx < 2:
        keep_inner_layers_range = None
    else:
        keep_inner_layers_range = (0, target_level_idx - 2)
    model = FrozenPartialPLM(
        plm_name=args.plm_name,
        keep_embed_layer=keep_embed_layer,
        keep_inner_layers_range=keep_inner_layers_range)

    inferencer = Trainer(
        accelerator='gpu',
        devices=args.devices,
        precision=args.precision,
        strategy='ddp' if len(args.devices) > 1 else None,
        callbacks=[
            PredictWriter(output_dir=args.processed_dir,
                          plm_name=args.plm_name,
                          n_freeze_layers=model.n_freeze_layers,
                          n_unfreeze_layers=args.plm_last_n_unfreeze,
                          write_interval="epoch")
        ],
        logger=False,
        deterministic=True,
        enable_model_summary=True,
    )
    inferenced_data = inferencer.predict(model, items_loader)