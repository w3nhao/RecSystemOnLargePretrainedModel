import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datamodules.datamodule import DataModule
from datamodules.utils import (ratio_split, str_fields2ndarray,
                               point_wise_leave_one_out_split, 
                               ITEM_ID_SEQ_FIELD, TEXT_ID_SEQ_FIELD,
                               USER_ID_FIELD, ATTENTION_MASK_FIELD)
from datamodules.data_preprocessor import DataPreprocessor
from datamodules.dataset import  TextPointWiseRecDataset
from datamodules.configs import PointWiseDataModuleConfig
from datamodules.utils import GLOBAL_RANDOM_SEED

from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PointWiseDataModule(DataModule):

    def __init__(self, dm_config: PointWiseDataModuleConfig):
        self.save_hyperparameters(logger=True)
        super().__init__(dm_config)
        max_len = dm_config.max_item_seq_len
        max_len = max_len if max_len else "INF"
        self.processed_dir = os.path.join(
            self.data_configs["data_dir"],
            f"{dm_config.dataset}"
            f"_maxlen@{max_len}"
            f"_minlen@{dm_config.min_item_seq_len}"
            f"_toklen@{dm_config.tokenized_len}"
            "_processed")
        
        sampling_n = dm_config.sampling_n
        
        item_table = self.data_configs["item_table"]
        item_file = f"{item_table}_{self.tokenizer_abbr}.processed.tsv"
        self.items_path = os.path.join(self.processed_dir, item_file)
        
        inter_table = self.data_configs["inter_table"]
        inter_file = f"{inter_table}.processed.tsv"
        self.inters_path = os.path.join(self.processed_dir, inter_file)
        
        if sampling_n is not None:
            self.resample_dir = os.path.join(
                self.processed_dir, f"point_wise_resample_n@{sampling_n}")
            
            self.sampled_inters_path = os.path.join(
                self.resample_dir, inter_file)
            self.sampled_items_path = os.path.join(
                self.resample_dir, item_file)
            
            sampled_iids_file = f"resampled@{sampling_n}_iids.tsv"
            self.sampled_iids_path = os.path.join(
                self.resample_dir, sampled_iids_file)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        max_item_seq_len = self.hparams.dm_config.max_item_seq_len
        min_item_seq_len = self.hparams.dm_config.min_item_seq_len
        tokenized_len = self.hparams.dm_config.tokenized_len
        plm_name = self.hparams.dm_config.plm_name
        sampling_n = self.hparams.dm_config.sampling_n

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
            
        else:
            inters_exist = os.path.isfile(self.inters_path)
            items_exist = os.path.isfile(self.items_path)
            
            if not inters_exist or not items_exist:
                data_prep.prepare_data()
            
            if not inters_exist:
                log.warning(f"Inters not exist in {self.processed_dir}")
                data_prep.prepare_point_wise_inters()
                data_prep.save_inters(self.processed_dir)
            
            if not items_exist:
                log.warning(f"Items not exist in {self.items_path}")
                data_prep.prepare_items(
                    plm_name=plm_name,
                    tokenized_len=tokenized_len)
                data_prep.save_items(self.processed_dir, self.tokenizer_abbr)
                
        inters = pd.read_csv(self.inters_path, sep="\t", header=0)
        items = pd.read_csv(self.items_path, sep="\t", header=0)
        
        num_items = len(items)
        
        # if resampling is needed
        if sampling_n is not None:
            os.makedirs(self.resample_dir, exist_ok=True)
            assert sampling_n < len(inters), \
                "sampling_n should be smaller than the number of interactions"
            
            # sampling interactions
            inters = inters.sample(n=sampling_n, random_state=GLOBAL_RANDOM_SEED)
            user_ids = inters[USER_ID_FIELD].values
            item_id_seqs = str_fields2ndarray(
                df=inters, fields=[ITEM_ID_SEQ_FIELD], field_lens=[None])
            
            # sampling items
            sampled_iids = np.sort(np.unique(np.reshape(item_id_seqs, -1)))
            
            # remap item ids
            new_items = items.iloc[sampled_iids].reset_index(drop=True)
            old_iids_new_iids = np.empty(num_items, dtype=np.int64)
            new_iids = np.arange(len(sampled_iids))
            np.put_along_axis(old_iids_new_iids, sampled_iids, new_iids, axis=0)
            
            new_iid_seqs = old_iids_new_iids[sampled_iids]
            
            new_iid_seqs = np.array(
                [" ".join(seq) for seq in new_iid_seqs.astype(str)])
            
            # save resampled interactions and items
            new_inters = pd.DataFrame({
                USER_ID_FIELD: user_ids,
                ITEM_ID_SEQ_FIELD: new_iid_seqs,
            })
            
            new_inters.to_csv(self.sampled_inters_path, sep="\t", index=False)
            new_items.to_csv(self.sampled_items_path, sep="\t", index=False)
            
            # save item id mapping
            sampled_iids_df = pd.DataFrame(sampled_iids)
            sampled_iids_df.to_csv(
                self.sampled_iids_path, sep="\t", index=False, header=False)
            
            num_items = len(new_items)
        
        return num_items
    
    
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            sampling_n = self.hparams.dm_config.sampling_n
            if sampling_n is not None:
                inters_path = self.sampled_inters_path
                items_path = self.sampled_items_path
            else:
                inters_path = self.inters_path
                items_path = self.items_path
                
            inters = pd.read_csv(inters_path, sep="\t", header=0)
            items = pd.read_csv(items_path, sep="\t", header=0)
            
            log.info(f"Number of behaviors: {len(inters)}")
            log.info(f"Number of items: {len(items)}")
            
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

            tokenized_ids, attention_mask = str_fields2ndarray(
                df=items,
                fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
                field_lens=[tokenized_len, tokenized_len],
            )
            
            # split data
            stages = ["train", "val", "test"]
            split_type = self.hparams.dm_config.split_type
            user_ids, item_id_seqs = self._split_processed_inters_df(
                inters=inters, split_type=split_type, stages=stages)

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
            
            self.num_items = len(items)
    
    def _split_processed_inters_df(self,
                                   inters,
                                   split_type="ratio",
                                   stages=["train", "val", "test"]):
        split_type = self.hparams.dm_config.split_type

        user_ids, item_id_seqs = {}, {}
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

        return user_ids, item_id_seqs

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
            sampling_n=args.sampling_n,
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