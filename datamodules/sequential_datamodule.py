import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datamodules.datamodule import DataModule
from datamodules.utils import (pre_inference, ratio_split,
                               seq_leave_one_out_split, str_fields2ndarray,
                               ITEM_ID_SEQ_FIELD, TARGET_FIELD,
                               TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD,
                               InferenceFileProcessor)
from datamodules.data_preprocessor import DataPreprocessor
from datamodules.dataset import TextSeqRecDataset, PreInferTextSeqRecDataset 
from datamodules.configs import PreInferSeqDataModuleConfig, SeqDataModuleConfig

from utils.cli_parse import parse_boolean
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SeqDataModule(DataModule):

    def __init__(self, dm_config: SeqDataModuleConfig):
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)
        super().__init__(dm_config)
        max_len = dm_config.max_item_seq_len if dm_config.max_item_seq_len else "INF"
        self.processed_dir = os.path.join(
            self.data_configs["data_dir"],
            f"{dm_config.dataset}"
            f"_maxlen@{max_len}"
            f"_minlen@{dm_config.min_item_seq_len}"
            f"_toklen@{dm_config.tokenized_len}"
            "_processed",
        )
        self.inters_save_dir = os.path.join(
            self.processed_dir,
            f"inters_saslen@{dm_config.sasrec_seq_len}")
        
    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        max_item_seq_len = self.hparams.dm_config.max_item_seq_len
        min_item_seq_len = self.hparams.dm_config.min_item_seq_len
        tokenized_len = self.hparams.dm_config.tokenized_len
        sasrec_seq_len = self.hparams.dm_config.sasrec_seq_len
        plm_name = self.hparams.dm_config.plm_name
        sampling_n = self.hparams.dm_config.sampling_n

        data_prep = DataPreprocessor(
            data_cfg=self.data_configs,
            max_item_seq_len=max_item_seq_len,
            min_item_seq_len=min_item_seq_len,
        )

        if not os.path.exists(self.processed_dir):
            data_prep.prepare_data()
            data_prep.prepare_seq_inters(sasrec_seq_len=sasrec_seq_len)
            data_prep.prepare_items(
                plm_name=plm_name,
                tokenized_len=tokenized_len,
            )

            os.makedirs(self.processed_dir)
            os.makedirs(self.inters_save_dir)
            
            data_prep.save_inters(self.inters_save_dir)
            data_prep.save_items(self.processed_dir, self.tokenizer_abbr)
            
            num_items = data_prep.num_items
        else:
            item_table = self.data_configs["item_table"]
            item_file = f"{item_table}_{self.tokenizer_abbr}.processed.tsv"
            items_path = os.path.join(self.processed_dir, item_file)
            
            inters_exist = os.path.exists(self.inters_save_dir)
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
                    log.warning(f"Inters not exist in {self.inters_save_dir}")
                    data_prep.prepare_seq_inters(sasrec_seq_len=sasrec_seq_len)
                    os.makedirs(self.inters_save_dir)
                    data_prep.save_inters(self.inters_save_dir)
                    
                num_items = data_prep.num_items
                
        if sampling_n is not None:
            assert sampling_n < len(inters)
        
        return num_items

    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            tokenized_len = self.hparams.dm_config.tokenized_len
            sasrec_seq_len = self.hparams.dm_config.sasrec_seq_len

            inter_table = self.data_configs["inter_table"]
            item_table = self.data_configs["item_table"]
            inters_path = os.path.join(self.inters_save_dir,
                                       f"{inter_table}.processed.tsv")
            items_path = os.path.join(
                self.processed_dir,
                f"{item_table}_{self.tokenizer_abbr}.processed.tsv")
            inters = pd.read_csv(inters_path, sep="\t", header=0)
            items = pd.read_csv(items_path, sep="\t", header=0)

            self.num_items = len(items)
            
            tokenized_ids, attention_mask = str_fields2ndarray(
                df=items,
                fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
                field_lens=[tokenized_len, tokenized_len],
            )

            stages = ["train", "val", "test"]
            input_item_id_seqs, target_item_id_seqs = {}, {}
            split_type = self.hparams.dm_config.split_type
            if split_type == "ratio":
                log.info("Splitting data by ratio: train/val/test = 0.8/0.1/0.1")
                splitted_df = ratio_split(data=inters, ratios=[0.8, 0.1, 0.1])
                for df, stage in zip(splitted_df, stages):
                    input_item_id_seqs[stage], target_item_id_seqs[stage] = \
                        str_fields2ndarray(
                            df=df,
                            fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
                            field_lens=[sasrec_seq_len, sasrec_seq_len],
                        )
            elif split_type == "leave_one_out":
                log.info("Splitting data by leave-one-out method")
                input_seqs, target_seqs = str_fields2ndarray(
                    df=inters,
                    fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
                    field_lens=[sasrec_seq_len, sasrec_seq_len],
                )
                splitted_data = seq_leave_one_out_split(
                    input_seqs=input_seqs,
                    targets=target_seqs)
                for _data, stage in zip(splitted_data, stages):
                    input_item_id_seqs[stage], target_item_id_seqs[stage] = _data
            else:
                raise ValueError(f"Unknown split type: {split_type}")

            [data_train, data_val, data_test] = [
                TextSeqRecDataset(
                    input_id_seqs=input_item_id_seqs[stage],
                    target_id_seqs=target_item_id_seqs[stage],
                    tokenized_ids=tokenized_ids,
                    attention_mask=attention_mask,
                )
                for stage in stages
            ]

            self.data_train = data_train
            self.data_val = data_val
            self.data_test = data_test
    
    # TODO
    def _downsampling(self, inters_df, items_df, downsample_num):
        """Downsample data.
        """
        pass

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

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser):
        """Add datamodule specific arguments to the parser."""
        parent_parser = super().add_datamodule_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("SeqRecDataModule")
        parser.add_argument("--sasrec_seq_len", type=int, default=20)
        return parent_parser

    @classmethod
    def build_datamodule_config(cls, args):
        """Build configs from arguments."""
        config = SeqDataModuleConfig(
            dataset=args.dataset,
            split_type=args.split_type,
            sampling_n=args.sampling_n,
            min_item_seq_len=args.min_item_seq_len,
            max_item_seq_len=args.max_item_seq_len,
            sasrec_seq_len=args.sasrec_seq_len,
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


class PreInferSeqDataModule(SeqDataModule):

    def __init__(self, dm_config: PreInferSeqDataModuleConfig):
        super().__init__(dm_config)

    def prepare_data(self):
        num_items = super().prepare_data()

        tokenized_len = self.hparams.dm_config.tokenized_len
        # save tokenized_ids and attention_mask as npy
        new_item_file = f"{item_table}_{self.tokenizer_abbr}.processed.npy"
        new_items_path = os.path.join(self.processed_dir, new_item_file)
        if not os.path.isfile(new_items_path):
            # load old preprocessed tokenized_ids and attention_mask
            item_table = self.data_configs["item_table"]
            item_file = f"{item_table}_{self.tokenizer_abbr}.processed.tsv"
            items_path = os.path.join(self.processed_dir, item_file)
            items = pd.read_csv(items_path, sep="\t", header=0)
            tokenized_ids, attention_mask = str_fields2ndarray(
                df=items,
                fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
                field_lens=[tokenized_len, tokenized_len],
            )
            tokenized_ids = np.expand_dims(tokenized_ids, axis=0)
            attention_mask = np.expand_dims(attention_mask, axis=0)
            items = np.concatenate((tokenized_ids, attention_mask), axis=0)
            np.save(new_items_path, items)
            
        plm_name = self.hparams.dm_config.plm_name
        last_n_unfreeze = self.hparams.dm_config.plm_last_n_unfreeze
        pre_inference_batch_size = self.hparams.dm_config.pre_inference_batch_size
        pre_inference_devices = self.hparams.dm_config.pre_inference_devices
        pre_inference_precision = self.hparams.dm_config.pre_inference_precision
        pre_inference_num_workers = self.hparams.dm_config.pre_inference_num_workers
        pre_inference_layer_wise = self.hparams.dm_config.pre_inference_layer_wise
        
        pre_inference(
            plm_name=plm_name,
            last_n_unfreeze=last_n_unfreeze,
            processed_dir=self.processed_dir,
            item_file=item_file,
            tokenized_len=tokenized_len,
            batch_size=pre_inference_batch_size,
            devices=pre_inference_devices,
            precision=pre_inference_precision,
            num_workers=pre_inference_num_workers,
            layer_wise=pre_inference_layer_wise,
        )
        
        return num_items
    
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            sasrec_seq_len = self.hparams.dm_config.sasrec_seq_len
            plm_name = self.hparams.dm_config.plm_name
            last_n_unfreeze = self.hparams.dm_config.plm_last_n_unfreeze
            
            inter_table = self.data_configs["inter_table"]
            inters_file = f"{inter_table}.processed.tsv"
            inters_path = os.path.join(self.inters_save_dir, inters_file)
            inters = pd.read_csv(inters_path, sep="\t", header=0)

            stages = ["train", "val", "test"]
            input_item_id_seqs, target_item_id_seqs = {}, {}
            split_type = self.hparams.dm_config.split_type
            if split_type == "ratio":
                log.info("Splitting data by ratio: train/val/test = 0.8/0.1/0.1")
                splitted_df = ratio_split(data=inters, ratios=[0.8, 0.1, 0.1])
                for df, stage in zip(splitted_df, stages):
                    input_item_id_seqs[stage], target_item_id_seqs[stage] = \
                        str_fields2ndarray(
                            df=df,
                            fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
                            field_lens=[sasrec_seq_len, sasrec_seq_len],
                        )
            elif split_type == "leave_one_out":
                log.info("Splitting data by leave-one-out method")
                input_seqs, target_seqs = str_fields2ndarray(
                    df=inters,
                    fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
                    field_lens=[sasrec_seq_len, sasrec_seq_len],
                )
                splitted_data = seq_leave_one_out_split(
                    input_seqs=input_seqs,
                    targets=target_seqs)
                for _data, stage in zip(splitted_data, stages):
                    input_item_id_seqs[stage], target_item_id_seqs[stage] = _data
            else:
                raise ValueError(f"Unknown split type: {split_type}")

            item_table = self.data_configs["item_table"]
            item_file = f"{item_table}_{self.tokenizer_abbr}.processed.npy"
            items_path = os.path.join(self.processed_dir, item_file)
            items = np.load(items_path, allow_pickle=True)
            attention_mask = items[1]

            file_processor = InferenceFileProcessor(
                        processed_dir=self.processed_dir,
                        plm_name=plm_name,
                        last_n_unfreeze=last_n_unfreeze,
                    )
            
            tokenized_embs_file = file_processor.get_inference_file()
            tokenized_embs_path = os.path.join(self.processed_dir, tokenized_embs_file)
            tokenized_embs = torch.load(tokenized_embs_path)

            self.num_items = len(items)

            [data_train, data_val, data_test] = [
                PreInferTextSeqRecDataset(
                    input_id_seqs=input_item_id_seqs[stage],
                    target_id_seqs=target_item_id_seqs[stage],
                    tokenized_embs=tokenized_embs,
                    attention_mask=attention_mask,
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
        parser = parent_parser.add_argument_group("PreInferSeqDataModule")
        parser.add_argument("--pre_inference_devices",
                            type=int,
                            nargs="+",
                            default=[0, 1, 2, 3, 4, 5, 6, 7])
        parser.add_argument("--pre_inference_precision", type=int, default=32)
        parser.add_argument("--pre_inference_batch_size", type=int, default=1)
        parser.add_argument("--pre_inference_num_workers", type=int, default=4)
        parser.add_argument("--pre_inference_layer_wise", type=parse_boolean, default=False)
        return parent_parser

    @classmethod
    def build_datamodule_config(cls, args):
        """Build configs from arguments."""
        config = PreInferSeqDataModuleConfig(
            dataset=args.dataset,
            split_type=args.split_type,
            sampling_n=args.sampling_n,
            plm_name=args.plm_name,
            plm_last_n_unfreeze=args.plm_last_n_unfreeze,
            pre_inference_batch_size=args.pre_inference_batch_size,
            pre_inference_precision=args.pre_inference_precision,
            pre_inference_devices=args.pre_inference_devices,
            pre_inference_num_workers=args.pre_inference_num_workers,
            pre_inference_layer_wise=args.pre_inference_layer_wise,
            min_item_seq_len=args.min_item_seq_len,
            max_item_seq_len=args.max_item_seq_len,
            sasrec_seq_len=args.sasrec_seq_len,
            tokenized_len=args.tokenized_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        return config
