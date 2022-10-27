import numpy as np
import torch
from torch.utils.data import Dataset
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class IDSeqRecDataset(Dataset):
    def __init__(
        self,
        input_id_seqs,
        target_id_seqs,
        padding_idx=0,
    ):
        self._len = len(input_id_seqs)
        self.padding_idx = padding_idx
        self.input_id_seqs = input_id_seqs
        self.target_id_seqs = target_id_seqs
        self.item_seq_masks = self._get_masks(self.input_id_seqs)

    def _get_masks(self, data):
        masks = np.where(data != self.padding_idx, True, False)
        return masks
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        item_id_seq = self.input_id_seqs[idx]
        target_id_seq = self.target_id_seqs[idx]
        item_seq_mask = self.item_seq_masks[idx]
        return target_id_seq, item_id_seq, item_seq_mask


class TextSeqRecDataset(IDSeqRecDataset):

    def __init__(
        self,
        input_id_seqs,
        target_id_seqs,
        tokenized_ids,
        attention_mask,
        padding_idx=0,
    ):
        super().__init__(input_id_seqs, target_id_seqs, padding_idx)
        self.tokenized_ids = tokenized_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        target_id_seq, item_id_seq, item_seq_mask = super().__getitem__(idx)
        tokenized_ids = self.tokenized_ids[item_id_seq]
        attention_mask = self.attention_mask[item_id_seq]
        return target_id_seq, item_id_seq, item_seq_mask, \
            tokenized_ids, attention_mask


class PreInferTextSeqRecDataset(IDSeqRecDataset):
  
    def __init__(
        self,
        input_id_seqs,
        target_id_seqs,
        tokenized_embs,
        attention_mask,
        padding_idx=0,
    ):
        super().__init__(input_id_seqs, target_id_seqs, padding_idx)
        self.tokenized_embs = tokenized_embs
        self.attention_mask = attention_mask

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        target_id_seq, item_id_seq, item_seq_mask = super().__getitem__(idx)
        tokenized_embs = self.tokenized_embs[item_id_seq]
        attention_mask = self.attention_mask[item_id_seq]
        return target_id_seq, item_id_seq, item_seq_mask, \
            tokenized_embs, attention_mask


class AllFreezePreInferTextSeqRecDataset(IDSeqRecDataset):
  
    def __init__(
        self,
        input_id_seqs,
        target_id_seqs,
        item_embs,
        padding_idx=0,
    ):
        super().__init__(input_id_seqs, target_id_seqs, padding_idx)
        self.item_embs = item_embs
        
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        target_id_seq, item_id_seq, item_seq_mask = super().__getitem__(idx)
        item_emb_seq = self.item_embs[item_id_seq]
        return target_id_seq, item_id_seq, item_seq_mask, item_emb_seq


class IDPointWiseRecDataset(Dataset):
    def __init__(
        self,
        n_items, 
        user_ids, 
        item_id_seqs, 
        n_neg_sampling=1
        ):
        self.n_items = n_items
        self.n_sampling = n_neg_sampling
        self.user_ids, self.item_ids = self._flatten(user_ids, item_id_seqs)
        self._len = len(self.user_ids)
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        return uid, iid

    def _flatten(self, user_ids, item_id_seqs):
        extended_user_ids = np.empty_like(item_id_seqs)
        for i, item_id_seq in enumerate(item_id_seqs):
            extended_user_ids[i] = np.full_like(item_id_seq, user_ids[i])
        uids = np.concatenate(extended_user_ids, axis=0)
        iids = np.concatenate(item_id_seqs, axis=0)
        return uids, iids
    
    def collect_fn(self, user_ids, item_ids):
        user_ids = torch.stack(user_ids, dim=0)
        item_ids = torch.stack(item_ids, dim=0)
        
        neg_samples = torch.randint(0, self.n_items, (self.n_sampling, *item_ids.shape))
        # mark which sample is positive
        pos_mask = (neg_samples == item_ids.unsqueeze(0))
        while pos_mask.any():
            neg_samples[pos_mask] = torch.randint(0, self.n_items, (pos_mask.sum(),))
            pos_mask = (neg_samples == item_ids.unsqueeze(0))
            
        return user_ids, item_ids, neg_samples
    
    
class TextPointWiseRecDataset(IDPointWiseRecDataset):
    def __init__(
        self, 
        n_items, 
        user_ids, 
        item_id_seqs,
        tokenized_ids,
        attention_mask,
        n_neg_sampling=1
        ):
        super().__init__(n_items, user_ids, item_id_seqs, n_neg_sampling)
        self.tokenized_ids = tokenized_ids
        self.attenion_mask = attention_mask
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        return uid, iid, self.tokenized_ids[iid], self.attenion_mask[iid]

    def _flatten(self, user_ids, item_id_seqs):
        extended_user_ids = np.empty_like(item_id_seqs)
        for i, item_id_seq in enumerate(item_id_seqs):
            extended_user_ids[i] = np.full_like(item_id_seq, user_ids[i])
        uids = np.concatenate(extended_user_ids, axis=0)
        iids = np.concatenate(item_id_seqs, axis=0)
        return uids, iids
        
    def collect_fn(self, user_ids, item_ids, tokenized_ids, attention_mask):
        user_ids = torch.stack(user_ids, dim=0)
        item_ids = torch.stack(item_ids, dim=0)
        tokenized_ids = torch.stack(tokenized_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        
        neg_samples = torch.randint(0, self.n_items, (self.n_sampling, *item_ids.shape))
        # mark which sample is positive
        pos_mask = (neg_samples == item_ids.unsqueeze(0))
        while pos_mask.any():
            neg_samples[pos_mask] = torch.randint(0, self.n_items, (pos_mask.sum(),))
            pos_mask = (neg_samples == item_ids.unsqueeze(0))
            
        neg_tokenized_ids = self.tokenized_ids[neg_samples]
        neg_attention_mask = self.attenion_mask[neg_samples]
        
        return user_ids, item_ids, tokenized_ids, attention_mask, \
            neg_samples, neg_tokenized_ids, neg_attention_mask

