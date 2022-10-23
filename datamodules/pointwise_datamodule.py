import os
import pandas as pd
from torch.utils.data import DataLoader
from datamodules.datamodule import DataModule
from datamodules.utils import (ratio_split, str_fields2ndarray,
                               point_wise_leave_one_out_split, 
                               ITEM_ID_SEQ_FIELD, TEXT_ID_SEQ_FIELD,
                               USER_ID_FIELD, ATTENTION_MASK_FIELD)
from datamodules.data_preprocessor import DataPreprocessor
from datamodules.dataset import  TextPointWiseRecDataset
from datamodules.configs import PointWiseDataModuleConfig

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PointWiseDataModule(DataModule):

    def __init__(self, dm_config: PointWiseDataModuleConfig):
        super().__init__(dm_config)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        max_item_seq_len = self.hparams.dm_config.max_item_seq_len
        min_item_seq_len = self.hparams.dm_config.min_item_seq_len
        tokenized_len = self.hparams.dm_config.tokenized_len
        plm_name = self.hparams.dm_config.plm_name

        data_prep = DataPreprocessor(
            data_cfg=self.data_configs,
            max_item_seq_len=max_item_seq_len,
            min_item_seq_len=min_item_seq_len,
        )

        if not os.path.exists(self.processed_dir):
            data_prep.prepare_data()
            data_prep.prepare_point_wise_inters()
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
            item_file = f"{item_table}_{self.tokenizer_abbr}.processed.tsv"
            items_path = os.path.join(self.processed_dir, item_file)
            
            inter_table = self.data_configs["inter_table"]
            inter_file = f"{inter_table}.processed.tsv"
            inter_path = os.path.join(self.processed_dir, inter_file)
            
            inters_exist = os.path.isfile(inter_path)
            items_exist = os.path.isfile(items_path)
            
            if items_exist:
                items = pd.read_csv(items_path, sep="\t", header=0)
                num_items = len(items)
            else:
                data_prep.prepare_data()
                log.warning(f"Items not exist in {items_path}")
                data_prep.prepare_items(
                    plm_name=plm_name,
                    tokenized_len=tokenized_len,
                )
                data_prep.save_items(self.processed_dir, self.tokenizer_abbr)
                
                if not inters_exist:
                    log.warning(f"Inters not exist in {inter_path}")
                    data_prep.prepare_point_wise_inters()
                    data_prep.save_inters(self.processed_dir)
                    
                num_items = data_prep.num_items
        
        return num_items
    
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            tokenized_len = self.hparams.dm_config.tokenized_len
            n_neg_sampling = self.hparams.dm_config.n_neg_sampling

            inter_table = self.data_configs["inter_table"]
            item_table = self.data_configs["item_table"]
            inters_path = os.path.join(self.processed_dir,
                                       f"{inter_table}.processed.tsv")
            items_path = os.path.join(
                self.processed_dir,
                f"{item_table}_{self.tokenizer_abbr}.processed.tsv")
            inters = pd.read_csv(inters_path, sep="\t", header=0)
            items = pd.read_csv(items_path, sep="\t", header=0)

            # TODO: downsample

            self.num_items = len(items)

            tokenized_ids, attention_mask = str_fields2ndarray(
                df=items,
                fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
                field_lens=[tokenized_len, tokenized_len],
            )

            stages = ["train", "val", "test"]
            user_ids, item_id_seqs = {}, {}
            split_type = self.hparams.dm_config.split_type
            if split_type == "ratio":
                log.info("Splitting data by ratio: train/val/test = 0.8/0.1/0.1")
                splitted_df = ratio_split(data=inters, ratios=[0.8, 0.1, 0.1])
                for df, stage in zip(splitted_df, stages):
                    user_ids[stage] = df[USER_ID_FIELD].values
                    item_id_seqs[stage] = \
                        str_fields2ndarray(df=df, fields=[ITEM_ID_SEQ_FIELD],
                                           field_lens=[None])
            elif split_type == "leave_one_out":
                log.info("Splitting data by leave-one-out method")
                uids = df[USER_ID_FIELD].values
                iid_seqs = str_fields2ndarray(
                    df=inters,
                    fields=[ITEM_ID_SEQ_FIELD],
                    field_lens=[None],
                )
                splitted_data = point_wise_leave_one_out_split(
                    user_ids=uids,
                    item_id_seqs=iid_seqs)
                for _data, stage in zip(splitted_data, stages):
                    user_ids[stage], item_id_seqs[stage] = _data
            else:
                raise ValueError(f"Unknown split type: {split_type}")

            [data_train, data_val, data_test] = [
                TextPointWiseRecDataset(
                    n_items=self.num_items,
                    user_ids=user_ids[stage],
                    item_id_seqs=item_id_seqs[stage],
                    tokenized_ids=tokenized_ids,
                    attention_mask=attention_mask,
                    n_neg_sampling=n_neg_sampling,
                )
                for stage in stages
            ]

            self.data_train = data_train
            self.data_val = data_val
            self.data_test = data_test
    

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.dm_config.batch_size,
            num_workers=self.hparams.dm_config.num_workers,
            pin_memory=self.hparams.dm_config.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collect_fn,
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

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser):
        """Add datamodule specific arguments to the parser."""
        parent_parser = super().add_datamodule_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("PointWiseRecDataModule")
        parser.add_argument("--n_neg_sampling", type=int, default=1)
        return parent_parser

    @classmethod
    def build_datamodule_config(cls, args):
        """Build configs from arguments."""
        config = PointWiseDataModuleConfig(
            dataset=args.dataset,
            split_type=args.split_type,
            min_item_seq_len=args.min_item_seq_len,
            max_item_seq_len=args.max_item_seq_len,
            n_neg_sampling=args.n_neg_sampling,
            tokenized_len=args.tokenized_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        try:
            config.plm_name = args.plm_name
        except AttributeError:
            log.warning(
                f"No plm_name in args, use default tokenizer:"
                f"'{config.plm_name}' to process text."
            )
        try:
            config.plm_last_n_unfreeze = args.plm_last_n_unfreeze
        except AttributeError:
            log.warning(
                f"No plm_last_n_unfreeze in args, "
                f"use default value: {config.plm_last_n_unfreeze}."
            )
        return config