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
from datamodules.utils import PRETRAIN_MODEL_ABBR, InferenceFileProcessor
from models.partial_opt import PartialOPTModel

from transformers import logging
logging.set_verbosity_error()

from utils.pylogger import get_pylogger
log = get_pylogger(__name__)

class TokenizedDataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

    def __len__(self):
        return len(self.input_ids)
    

class EmbeddingsDataset(TokenizedDataset):

    def __init__(self, input_ids, attention_mask, inputs_hidden_state):
        super().__init__(input_ids, attention_mask)
        self.inputs_hidden_state = inputs_hidden_state
    
    def __getitem__(self, index):
        return self.input_ids[index], \
               self.attention_mask[index], \
               self.inputs_hidden_state[index]
    
    def __len__(self):
        return super().__len__()


class PredictWriter(BasePredictionWriter):

    def __init__(self, output_dir, plm_name,
                 last_n_unfreeze, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.plm_abbr = PRETRAIN_MODEL_ABBR[plm_name]
        self.last_n_unfreeze = last_n_unfreeze
        self.file_processor = InferenceFileProcessor(
            processed_dir=output_dir,
            plm_name=plm_name,
            last_n_unfreeze=self.last_n_unfreeze,
            )

    def write_on_epoch_end(self, trainer, pl_module, predictions,
                           batch_indices):
        rank_idx = pl_module.global_rank
        predictions = torch.cat(predictions[0], dim=0)
        batch_indices = [idx for idxs in batch_indices[0] for idx in idxs]
        batch_indices = torch.tensor(batch_indices)
        embs_file = self.file_processor.get_inference_file(is_embs=True,
                                                           rank_idx=rank_idx)
        idxs_file = self.file_processor.get_inference_file(is_embs=False,
                                                           rank_idx=rank_idx)
        torch.save(predictions, os.path.join(self.output_dir, embs_file))
        torch.save(batch_indices, os.path.join(self.output_dir, idxs_file))
        log.info(f"Rank {rank_idx} has written embeddings to {embs_file} and indices to {idxs_file}")

class FrozenPartialPLM(pl.LightningModule):

    def __init__(self,
                 plm_name,
                 keep_embed_layer,
                 keep_inner_layers_range,
                 ):
        """ keep inner layers range is a tuple of both inclusive index (start, end) """
        super(FrozenPartialPLM, self).__init__()
        self.plm_name = plm_name
        self.keep_embed_layer = keep_embed_layer
        self.keep_inner_layers_range = keep_inner_layers_range
        self.plm_config = AutoConfig.from_pretrained(self.plm_name)
        
        if self.plm_name.startswith("facebook"):
            self.plm = PartialOPTModel.from_pretrained(
                plm_name,
                keep_embed_layer=keep_embed_layer,
                keep_decoders_range=keep_inner_layers_range)

        elif self.plm_name.startswith("bert"):
            self.plm = BertModel.from_pretrained(plm_name)
            # if self.plm_last_n_unfreeze > 0:
            #     self.plm.encoder.layer = \
            #         self.plm.encoder.layer[:-self.plm_last_n_unfreeze]
            # else:
            #     self.plm.encoder.layer = self.plm.encoder.layer
            
        names = []
        params_num = 0
        for name, params in self.plm.named_parameters():
            params.requires_grad = False
            names.append(name)
            params_num += params.numel()
        log.info(f"Total memory of frozen parameters: {params_num * 4 / 1024 / 1024} MB")
        if len(names) > 4:
            log.info(f"Names of frozen parameters from {names[:2]} to {names[-2:]}")
        else:
            log.info(f"Names of frozen parameters: {names}")
        

    def forward(self, input_ids, attention_mask, inputs_hidden_state):
        output = self.plm(input_ids=input_ids,
                          attention_mask=attention_mask,
                          inputs_hidden_state=inputs_hidden_state)
        return output

    def predict_step(self, batch, batch_idx):
        if self.keep_embed_layer:
            input_ids, attention_mask = batch
            inputs_hidden_state = None
        else:
            _, attention_mask, inputs_hidden_state = batch
            input_ids = None
        output = self(input_ids, attention_mask, inputs_hidden_state)
        return output.last_hidden_state


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--plm_name", type=str, required=True)
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
    assert args.input_layer in levels[0:-1]
    assert args.target_layer in levels[1:]
    assert levels.index(args.input_layer) < levels.index(args.target_layer)
    
    input_layer = None if args.input_layer in ["none", "emb"] else args.input_layer 
    target_layer = None if args.target_layer == "emb" else args.target_layer 
    n_freeze = 0 if target_layer is None else args.target_layer + 1
    last_n_unfreeze = plm_config.num_hidden_layers - n_freeze
    
    # set the model
    keep_embed_layer = True if input_layer is None else False
    if target_layer is None:
        keep_inner_layers_range = None
    elif target_layer is not None and keep_embed_layer:
        keep_inner_layers_range = (0, target_layer)
    else:
        # the keep_decoders_range tuple is a left-closed and right-closed interval [a, b] as input 
        keep_inner_layers_range = (input_layer+1, target_layer)
    
    # load the processed input_ids and attention_mask
    items_path = os.path.join(args.processed_dir, args.processed_items_file)
    items = np.load(items_path, allow_pickle=True)
    input_ids = items[0]
    attention_mask = items[1]
    n_items = len(input_ids)
    
    # set the dataset
    # load the input item embeddings if needed
    if keep_embed_layer:
        items = TokenizedDataset(input_ids, attention_mask)
    else:
        assert args.input_item_embs_file is not None
        item_embs_path = os.path.join(args.processed_dir, args.input_item_embs_file)
        items_embs = torch.load(item_embs_path)
        # sanity check
        if len(items_embs) == n_items:
            items = EmbeddingsDataset(input_ids, attention_mask, items_embs)
        else:
            raise ValueError(
                "The number of items in collected inferenced data is not equal to the dataset's number of items." + \
                "This may becaused specifying a wrong inferenced embeddings file in `--input_item_embs_file`." + \
                "Or caused by the distributed inference process, which means data is sampled by the DDP sampler, some " + \
                "items may be sampled by multiple times to ensure the same workload of items for each device. " + \
                "Please use single device to inference if it is the case.")
            
    # set the dataloader
    items_loader = DataLoader(items,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
        
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
                          last_n_unfreeze=last_n_unfreeze,
                          write_interval="epoch"),
        ],
        logger=False,
        deterministic=True,
    )
    inferenced_data = inferencer.predict(model, items_loader)