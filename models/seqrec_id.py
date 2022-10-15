import torch
from utils.pylogger import get_pylogger
from utils.schedule_functions import get_lr_scheduler_function
from models.abstract_recommender import SeqRec
from models.configs import SeqRecConfig

log = get_pylogger(__name__)


class IDSeqRec(SeqRec):

    def __init__(self, config: SeqRecConfig):
        self.save_hyperparameters()
        super().__init__(self.hparams.config)

        # parameters initialization
        self.apply(self._init_weights)

    def _set_feature_extractor(self, config):
        self.item_embedding = torch.nn.Embedding(config.item_token_num,
                                                 config.sasrec_hidden_size,
                                                 padding_idx=0)

    def _feature_extract(self, item_id_seq, item_seq_mask, tokenized_ids,
                         attention_mask):
        item_embs = self.item_embedding(item_id_seq)
        return item_embs

    @classmethod
    def build_model_config(cls, args, item_token_num):
        config = SeqRecConfig(item_token_num=item_token_num)
        config = super(IDSeqRec, cls).build_model_config(args, config)
        return config
