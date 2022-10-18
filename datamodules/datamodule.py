import os
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datamodules.utils import (pre_inference, ratio_split, str_fields2ndarray,
                               ITEM_ID_SEQ_FIELD, TARGET_FIELD, TEXT_ID_SEQ_FIELD,
                               ATTENTION_MASK_FIELD, PRETRAIN_MODEL_ABBR)
from datamodules.data_preprocessor import DataPreprocessor
from datamodules.dataset import TextSeqRecDataset
from datamodules.configs import (PreInferSeqRecDMConfig, get_data_configs,
                                 SeqRecDataModuleConfig)

from utils.cli_parse import parse_boolean, int_or_none
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SeqDataModule(LightningDataModule):

    def __init__(self, dm_config: SeqRecDataModuleConfig):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        dataset = dm_config.dataset
        self.data_configs = get_data_configs(dataset)

        try:
            self.tokenizer_abbr = PRETRAIN_MODEL_ABBR[dm_config.plm_name]
        except KeyError:
            raise ValueError(f"Unsupport plm name: {dm_config.plm_name}")

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

        data_prep = DataPreprocessor(
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
            item_file = f"{item_table}_{self.tokenizer_abbr}.processed.tsv"
            items_path = os.path.join(self.processed_dir, item_file)
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
            inters_path = os.path.join(self.processed_dir,
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

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser):
        """Add datamodule specific arguments to the parser."""
        parser = parent_parser.add_argument_group("SeqRecDataModule")
        parser.add_argument("--dataset", type=str, default="MIND_small")
        parser.add_argument("--sasrec_seq_len", type=int, default=20)
        parser.add_argument("--tokenized_len", type=int, default=30)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--pin_memory", type=parse_boolean, default=False)
        parser.add_argument("--min_item_seq_len", type=int, default=5)
        parser.add_argument("--max_item_seq_len",
                            type=int_or_none,
                            default=None)
        return parent_parser

    @classmethod
    def build_datamodule_config(cls, args):
        """Build configs from arguments."""
        config = SeqRecDataModuleConfig(
            dataset=args.dataset,
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
            log.info(
                f"No plm_name in args, use default tokenizer:'{config.plm_name}' to process text."
            )
        try:
            config.plm_last_n_unfreeze = args.plm_last_n_unfreeze
        except AttributeError:
            log.info(
                f"No plm_last_n_unfreeze in args, use default value: {config.plm_last_n_unfreeze}."
            )
        return config


class PreInferSeqDataModule(SeqDataModule):

    def __init__(self, dm_config: PreInferSeqRecDMConfig):
        super().__init__(dm_config)

    def prepare_data(self):
        num_items = super().prepare_data()

        # load preprocessed tokenized_ids and attention_mask
        tokenized_len = self.hparams.dm_config.tokenized_len
        item_table = self.data_configs["item_table"]
        item_file = f"{item_table}_{self.tokenizer_abbr}.processed.tsv"
        items_path = os.path.join(self.processed_dir, item_file)
        items = pd.read_csv(items_path, sep="\t", header=0)
        self.num_items = len(items)
        tokenized_ids, attention_mask = str_fields2ndarray(
            df=items,
            fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
            field_len=tokenized_len,
        )

        # save tokenized_ids and attention_mask as npy
        item_file = f"{item_table}_{self.tokenizer_abbr}.processed.npy"
        items_path = os.path.join(self.processed_dir, item_file)
        if not os.path.isfile(items_path):
            tokenized_ids = np.expand_dims(tokenized_ids, axis=0)
            attention_mask = np.expand_dims(attention_mask, axis=0)
            items = np.concatenate((tokenized_ids, attention_mask), axis=0)
            np.save(items_path, items)
            
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

    # def setup(self, stage=None):
    #     # load and split datasets only if not loaded already
    #     if not self.data_train and not self.data_val and not self.data_test:
    #         tokenized_len = self.hparams.dm_config.tokenized_len
    #         sasrec_seq_len = self.hparams.dm_config.sasrec_seq_len
    #         last_n_unfreeze = self.hparams.dm_config.plm_last_n_unfreeze
    #         plm_name = self.hparams.dm_config.plm_name

    #         inter_table = self.data_configs["inter_table"]
    #         item_table = self.data_configs["item_table"]
    #         inters_path = os.path.join(self.processed_dir,
    #                                    f"{inter_table}.processed.tsv")
    #         items_path = os.path.join(
    #             self.processed_dir,
    #             f"{item_table}_{self.tokenizer_abbr}.processed.tsv")
    #         inters = pd.read_csv(
    #             inters_path,
    #             sep="\t",
    #             header=0,
    #         )
    #         items = pd.read_csv(
    #             items_path,
    #             sep="\t",
    #             header=0,
    #         )

    #         self.num_items = len(items)

    #         tokenized_ids, attention_mask = str_fields2ndarray(
    #             df=items,
    #             fields=[TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD],
    #             field_len=tokenized_len,
    #         )

    #         _, embs_file = inferenced_embs_exists(
    #             processed_dir=self.processed_dir,
    #             plm_name=plm_name,
    #             plm_last_n_unfreeze=last_n_unfreeze)
    #         pre_inference_embs = torch.load(
    #             os.path.join(self.processed_dir, embs_file))

    #         splitted_df = ratio_split(data=inters, ratios=[0.8, 0.1, 0.1])
    #         stages = ["train", "val", "test"]
    #         item_id_seqs, targets = {}, {}
    #         for df, stage in zip(splitted_df, stages):
    #             item_id_seqs[stage], targets[stage] = str_fields2ndarray(
    #                 df=df,
    #                 fields=[ITEM_ID_SEQ_FIELD, TARGET_FIELD],
    #                 field_len=sasrec_seq_len,
    #             )

    #         self.data_train = TextSeqRecDataset(
    #             item_id_seqs=item_id_seqs["train"],
    #             targets=targets["train"],
    #             tokenized_ids=tokenized_ids,
    #             attention_mask=attention_mask,
    #         )
    #         self.data_val = TextSeqRecDataset(
    #             item_id_seqs=item_id_seqs["val"],
    #             targets=targets["val"],
    #             tokenized_ids=tokenized_ids,
    #             attention_mask=attention_mask,
    #         )

    #         self.data_test = TextSeqRecDataset(
    #             item_id_seqs=item_id_seqs["test"],
    #             targets=targets["test"],
    #             tokenized_ids=tokenized_ids,
    #             attention_mask=attention_mask,
    #         )

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser):
        """Add datamodule specific arguments to the parser."""

        parser = parent_parser.add_argument_group("PreInferSeqRecDataModule")
        parser.add_argument("--dataset", type=str, default="MIND_small")
        parser.add_argument("--sasrec_seq_len", type=int, default=20)
        parser.add_argument("--tokenized_len", type=int, default=30)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--pin_memory", type=parse_boolean, default=False)
        parser.add_argument("--min_item_seq_len", type=int, default=5)
        parser.add_argument("--max_item_seq_len",
                            type=int_or_none,
                            default=None)
        parser.add_argument("--keep_n_freeze_files", type=int, nargs="+", default=None)
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
        config = PreInferSeqRecDMConfig(
            dataset=args.dataset,
            plm_name=args.plm_name,
            plm_last_n_unfreeze=args.plm_last_n_unfreeze,
            keep_n_freeze_files=args.keep_n_freeze_files,
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
