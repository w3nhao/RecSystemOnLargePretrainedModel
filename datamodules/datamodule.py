import os
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datamodules.utils import (
    DataPreporcessor,
    ratio_split,
    str_fields2ndarray,
    ITEM_ID_SEQ_FIELD,
    TARGET_FIELD,
    TEXT_ID_SEQ_FIELD,
    ATTENTION_MASK_FIELD,
    str_fields2ndarray,
)
from datamodules.dataset import TextSeqRecDataset
from datamodules.configs import get_configs

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)

PRETRAIN_MODEL_ABBR = {
    "facebook/opt-125m": "OPT125M",
    "facebook/opt-350m": "OPT350M",
    "facebook/opt-1.3b": "OPT1.3B",
    "facebook/opt-2.7b": "OPT2.7B",
    "facebook/opt-6.7b": "OPT6.7B",
    "facebook/opt-13b": "OPT13B",
    "facebook/opt-30b": "OPT30B",
}


class SeqDataModule(LightningDataModule):

    def __init__(
        self,
        data_name,
        min_item_seq_len,
        max_item_seq_len,
        pretrained_model,
        tokenized_len,
        sasrec_seq_len,
        batch_size,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        assert min_item_seq_len > 0 and tokenized_len > 0 and sasrec_seq_len > 0
        if max_item_seq_len is not None:
            assert max_item_seq_len > 0 and max_item_seq_len >= min_item_seq_len

        self.min_item_seq_len = min_item_seq_len
        self.max_item_seq_len = max_item_seq_len
        self.sasrec_seq_len = sasrec_seq_len
        self.tokenized_len = tokenized_len
        self.pretrained_model = pretrained_model

        self.data_configs = get_configs(data_name)

        encoder_abbr = PRETRAIN_MODEL_ABBR[self.pretrained_model]
        max_len = self.max_item_seq_len if self.max_item_seq_len else "INF"
        self.processed_dir = os.path.join(
            self.data_configs["data_dir"],
            f"{encoder_abbr}"
            f"_maxlen@{max_len}"
            f"_minlen@{self.min_item_seq_len}"
            f"_toklen@{self.tokenized_len}"
            f"_saslen@{self.sasrec_seq_len}"
            "_processed",
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        if not os.path.exists(self.processed_dir):

            data_prep = DataPreporcessor(
                data_cfg=self.data_configs,
                max_item_seq_len=self.max_item_seq_len,
                min_item_seq_len=self.min_item_seq_len,
            )

            data_prep.prepare_data()

            data_prep.prepare_inters(sasrec_seq_len=self.sasrec_seq_len)

            data_prep.prepare_items(
                pretrained_model=self.pretrained_model,
                tokenized_len=self.tokenized_len,
            )

            os.makedirs(self.processed_dir)
            data_prep.save_data(self.processed_dir)
            num_items = data_prep.num_items
        else:
            item_table = self.data_configs["item_table"]
            items = pd.read_csv(
                os.path.join(self.processed_dir,
                             f"{item_table}.processed.tsv"),
                sep="\t",
                header=0,
            )
            num_items = len(items)
        return num_items

    def setup(self, stage):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            inter_table = self.data_configs["inter_table"]
            item_table = self.data_configs["item_table"]

            inters = pd.read_csv(
                os.path.join(self.processed_dir,
                             f"{inter_table}.processed.tsv"),
                sep="\t",
                header=0,
            )
            items = pd.read_csv(
                os.path.join(self.processed_dir,
                             f"{item_table}.processed.tsv"),
                sep="\t",
                header=0,
            )

            self.num_items = len(items)

            tokenized_ids, attention_mask = str_fields2ndarray(
                df=items,
                fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
                field_len=self.tokenized_len,
            )

            splitted_df = ratio_split(data=inters, ratios=[0.8, 0.1, 0.1])
            stages = ["train", "val", "test"]
            item_id_seqs, targets = {}, {}
            for df, stage in zip(splitted_df, stages):
                item_id_seqs[stage], targets[stage] = str_fields2ndarray(
                    df=df,
                    fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
                    field_len=self.sasrec_seq_len,
                )

            self.data_train = TextSeqRecDataset(
                item_id_seqs=item_id_seqs["train"],
                targets=targets["train"],
                tokenized_ids=tokenized_ids,
                attention_mask=attention_mask,
            )

            self.data_val = TextSeqRecDataset(
                item_id_seqs=item_id_seqs["val"],
                targets=targets["val"],
                tokenized_ids=tokenized_ids,
                attention_mask=attention_mask,
            )

            self.data_test = TextSeqRecDataset(
                item_id_seqs=item_id_seqs["test"],
                targets=targets["test"],
                tokenized_ids=tokenized_ids,
                attention_mask=attention_mask,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict):
        """Things to do when loading checkpoint."""
        pass
