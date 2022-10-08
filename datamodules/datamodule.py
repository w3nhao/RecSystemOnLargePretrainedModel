import os
import pandas as pd
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
from datamodules.configs import get_data_configs, SeqRecDataModuleConfig

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
    "bert-base-uncased": "BERTBASE",
    "bert-large-uncased": "BERTLARGE",
}


class SeqDataModule(LightningDataModule):

    def __init__(self, dm_config: SeqRecDataModuleConfig):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        dataset = dm_config.dataset
        self.data_configs = get_data_configs(dataset)

        self.tokenizer_abbr = PRETRAIN_MODEL_ABBR[dm_config.plm_name]
        max_len = dm_config.max_item_seq_len if dm_config.max_item_seq_len else "INF"
        self.processed_dir = os.path.join(
            self.data_configs["data_dir"],
            f"{dataset}"
            f"_maxlen@{max_len}"
            f"_minlen@{dm_config.min_item_seq_len}"
            f"_toklen@{dm_config.tokenized_len}"
            f"_saslen@{dm_config.sasrec_seq_len}"
            "_processed",
        )

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        max_item_seq_len = self.hparams.dm_config.max_item_seq_len
        min_item_seq_len = self.hparams.dm_config.min_item_seq_len
        tokenized_len = self.hparams.dm_config.tokenized_len
        sasrec_seq_len = self.hparams.dm_config.sasrec_seq_len
        plm_name = self.hparams.dm_config.plm_name

        data_prep = DataPreporcessor(
            data_cfg=self.data_configs,
            max_item_seq_len=max_item_seq_len,
            min_item_seq_len=min_item_seq_len,
        )

        if not os.path.exists(self.processed_dir):
            data_prep.prepare_data()
            data_prep.prepare_inters(sasrec_seq_len=sasrec_seq_len)
            data_prep.prepare_items(
                plm_name=plm_name,
                tokenized_len=tokenized_len,
            )

            os.makedirs(self.processed_dir)
            data_prep.save_inters(self.processed_dir)
            data_prep.save_items(self.processed_dir, self.tokenizer_abbr)
            num_items = data_prep.num_items
        else:
            item_table = self.data_configs["item_table"]
            items_path = os.path.join(
                self.processed_dir,
                f"{item_table}_{self.tokenizer_abbr}.processed.tsv")
            if not os.path.isfile(items_path):
                data_prep.prepare_data()
                data_prep.prepare_items(
                    plm_name=plm_name,
                    tokenized_len=tokenized_len,
                )
                data_prep.save_items(self.processed_dir, self.tokenizer_abbr)
                num_items = data_prep.num_items
            else:
                items = pd.read_csv(
                    items_path,
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
        tokenized_len = self.hparams.dm_config.tokenized_len
        sasrec_seq_len = self.hparams.dm_config.sasrec_seq_len

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            inter_table = self.data_configs["inter_table"]
            item_table = self.data_configs["item_table"]
            inters_path = os.path.join(self.processed_dir,
                                       f"{inter_table}.processed.tsv")
            items_path = os.path.join(
                self.processed_dir,
                f"{item_table}_{self.tokenizer_abbr}.processed.tsv")
            inters = pd.read_csv(
                inters_path,
                sep="\t",
                header=0,
            )
            items = pd.read_csv(
                items_path,
                sep="\t",
                header=0,
            )

            self.num_items = len(items)

            tokenized_ids, attention_mask = str_fields2ndarray(
                df=items,
                fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
                field_len=tokenized_len,
            )

            splitted_df = ratio_split(data=inters, ratios=[0.8, 0.1, 0.1])
            stages = ["train", "val", "test"]
            item_id_seqs, targets = {}, {}
            for df, stage in zip(splitted_df, stages):
                item_id_seqs[stage], targets[stage] = str_fields2ndarray(
                    df=df,
                    fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
                    field_len=sasrec_seq_len,
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
        """Return the training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.dm_config.batch_size,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.dm_config.batch_size,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.dm_config.batch_size,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
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
