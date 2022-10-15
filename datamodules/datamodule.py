import os
import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datamodules.utils import (inferenced_embs_exists, ratio_split, str_fields2ndarray,
                               call_pre_inference, gather_inference_results,
                               ITEM_ID_SEQ_FIELD, TARGET_FIELD,
                               TEXT_ID_SEQ_FIELD, ATTENTION_MASK_FIELD,
                               PRETRAIN_MODEL_ABBR)
from datamodules.data_preprocessor import DataPreprocessor
from datamodules.dataset import TextSeqRecDataset
from datamodules.configs import get_data_configs, SeqRecDataModuleConfig

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

    def prepare_data(self, args):
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

        pre_inference_embs = None
        if args.input_type == "text" and \
            args.plm_n_unfreeze_layers > -1 and \
            args.pre_inference:
            already_inferenced, embs_file = inferenced_embs_exists(
                processed_dir=self.processed_dir,
                plm_name=plm_name,
                plm_n_unfreeze_layers=args.plm_n_unfreeze_layers
                )
            
            if already_inferenced:
                pre_inference_embs = torch.load(
                    os.path.join(self.processed_dir, embs_file)
                )
            else:
                return_code = call_pre_inference(
                    processed_dir=self.processed_dir,
                    item_file=item_file,
                    plm_name=plm_name,
                    plm_n_unfreeze_layers=args.plm_n_unfreeze_layers,
                    tokenized_len=tokenized_len,
                    batch_size=args.pre_inference_batch_size,
                    devices=args.pre_inference_devices,
                    precision=args.pre_inference_precision,
                )
                if return_code == 0:
                    pre_inference_embs = gather_inference_results(
                        processed_dir=self.processed_dir,
                        num_items=num_items,
                    )
                else:
                    raise RuntimeError("Pre-inference failed.")
        return pre_inference_embs, num_items

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

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser):
        """Add datamodule specific arguments to the parser."""

        def int_or_none(x):
            return None if x in ["none", "None", "NONE"] else int(x)

        parser = parent_parser.add_argument_group("SeqRecDataModule")
        parser.add_argument("--dataset", type=str, default="MIND_small")
        parser.add_argument("--sasrec_seq_len", type=int, default=20)
        parser.add_argument("--tokenized_len", type=int, default=30)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--pin_memory", type=bool, default=False)
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
            config.plm_n_unfreeze_layers = args.plm_n_unfreeze_layers
        except AttributeError:
            log.info(
                f"No plm_n_unfreeze_layers in args, use default value: {config.plm_n_unfreeze_layers}."
            )
        return config
