import numpy as np
from torch.utils.data import Dataset
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TextSeqRecDataset(Dataset):

    def __init__(
        self,
        item_id_seqs,
        targets,
        tokenized_ids,
        attention_mask,
        padding_idx=0,
    ):
        self._len = len(item_id_seqs)
        self.padding_idx = padding_idx
        self.item_id_seqs = item_id_seqs
        self.targets = targets
        self.item_seq_masks = self._get_masks(self.item_id_seqs)
        self.tokenized_ids = tokenized_ids
        self.attention_mask = attention_mask

    def _get_masks(self, data):
        masks = np.where(data != self.padding_idx, True, False)
        return masks

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        item_id_seq = self.item_id_seqs[idx]
        target_id_seq = self.targets[idx]
        item_seq_mask = self.item_seq_masks[idx]
        tokenized_ids = self.tokenized_ids[item_id_seq]
        attention_mask = self.attention_mask[item_id_seq]
        return item_id_seq, target_id_seq, item_seq_mask, tokenized_ids, attention_mask
