from pytorch_lightning import LightningDataModule
from datamodules.configs import DataModuleConfig, get_data_configs
from models.utils import PRETRAIN_MODEL_ABBR
from utils.cli_parse import parse_boolean, int_or_none
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DataModule(LightningDataModule):

    def __init__(self, dm_config: DataModuleConfig):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.data_configs = get_data_configs(dm_config.dataset)
        try:
            self.tokenizer_abbr = PRETRAIN_MODEL_ABBR[dm_config.plm_name]
        except KeyError:
            raise ValueError(f"Unsupport plm name: {dm_config.plm_name}")

        self.data_train = None
        self.data_val = None
        self.data_test = None

    @classmethod
    def add_datamodule_specific_args(cls, parent_parser):
        """Add datamodule specific arguments to the parser."""
        parser = parent_parser.add_argument_group("RecDataModule")
        parser.add_argument("--dataset", type=str, default="MIND_small")
        parser.add_argument("--sampling_n", type=int_or_none, default=None)
        parser.add_argument("--split_type", type=str, default="ratio")
        parser.add_argument("--tokenized_len", type=int, default=30)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--pin_memory", type=parse_boolean, default=False)
        parser.add_argument("--min_item_seq_len", type=int, default=5)
        parser.add_argument("--max_item_seq_len",
                            type=int_or_none,
                            default=None)
        return parent_parser


