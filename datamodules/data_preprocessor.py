import os
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, BertTokenizer
from datamodules.utils import (
    right_padding_left_trancate,
    tokenize,
    TEXT_ID_SEQ_FIELD,
    ATTENTION_MASK_FIELD,
)


class DataPreprocessor:

    def __init__(
        self,
        data_cfg,
        max_item_seq_len,
        min_item_seq_len,
    ) -> None:
        self.data_dir = data_cfg["data_dir"]
        self.uid_field = data_cfg["uid_field"]
        self.iid_field = data_cfg["iid_field"]
        self.item_text_field = data_cfg["item_text_field"]
        self.item_seq_field = data_cfg["item_seq_field"]
        self.inter_table = data_cfg["inter_table"]
        self.item_table = data_cfg["item_table"]
        self.table_configs = data_cfg["table_configs"]

        self._max_item_seq_len = max_item_seq_len
        self._min_item_seq_len = min_item_seq_len

        self.item_token_id, self.item_id_token, self.item_id_text = \
            None, None, None
            
        self.user_token_id, self.user_id_token = None, None

        self.lookup_df = self._load_data()
        self.processed_df = {}

    @property
    def num_items(self):
        return len(self.item_id_token)

    @property
    def min_item_seq_len(self):
        return self.lookup_df[self.inter_table][self.item_seq_field] \
            .apply(len).min()

    @property
    def max_item_seq_len(self):
        return self.lookup_df[self.inter_table][self.item_seq_field] \
            .apply(len).max()

    def prepare_seq_inters(self, sasrec_seq_len):
        """Prepare interactions data"""
        inters = self.lookup_df[self.inter_table][self.item_seq_field].values
        item_seqs, targets = right_padding_left_trancate(
            inters, sasrec_seq_len)

        item_seqs_str = np.array([" ".join(seq) for seq in item_seqs.astype(str)])
        targets_str = np.array([" ".join(seq) for seq in targets.astype(str)])

        self.processed_df[self.inter_table] = pd.DataFrame({
            "input_seqs": item_seqs_str,
            "targets": targets_str
        })
        
        return item_seqs, targets
    
    def prepare_point_wise_inters(self):
        """Prepare point-wise interactions data"""
        inters = self.lookup_df[self.inter_table].values
        user_ids = inters[:, 0]
        input_seqs = inters[:, 1]
        
        input_seqs_str = np.array([" ".join(seq) for seq in input_seqs.astype(str)])
         
        self.processed_df[self.inter_table] = pd.DataFrame({
            "user_id": user_ids,
            "input_seqs": input_seqs_str,
        })

    def prepare_items(self, plm_name, tokenized_len):
        """Prepare items data"""
        if plm_name.startswith("facebook/opt"):
            tokenizer = GPT2Tokenizer.from_pretrained(plm_name)
        elif plm_name.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(plm_name)
        else:
            raise NotImplementedError

        tokenized_seqs = tokenize(self.item_id_text, tokenizer, tokenized_len)

        tokenized_ids = tokenized_seqs[TEXT_ID_SEQ_FIELD].astype(np.str_)
        attention_mask = tokenized_seqs[ATTENTION_MASK_FIELD].astype(np.str_)

        tokenized_ids_str = np.array([" ".join(seq) for seq in tokenized_ids])
        attention_mask_str = np.array([" ".join(seq) for seq in attention_mask])

        self.processed_df[self.item_table] = pd.DataFrame({
            TEXT_ID_SEQ_FIELD:
            tokenized_ids_str,
            ATTENTION_MASK_FIELD:
            attention_mask_str,
        })
        
        return tokenized_ids, attention_mask

    def prepare_data(self):
        # self.drop_duplicates()
        self.item_token_id, self.item_id_token, self.item_id_text = \
            self._map_item_ID()
        self.user_token_id, self.user_id_token = self._map_user_ID()
        self._filter_item_seq_by_num(self._min_item_seq_len,
                                     self._max_item_seq_len)

    def _load_data(self):
        lookup_df = {}
        for table_name, cfg in self.table_configs.items():
            lookup_df[table_name] = pd.read_csv(
                cfg["filepath"],
                usecols=cfg["usecols"],
                dtype=cfg["filed_type"],
                delimiter="\t",
                header=0,
                encoding="utf-8",
                engine="python",
            )
            lookup_df[table_name].rename(columns=cfg["rename_cols"],
                                         inplace=True)

            if "token_seq_fields" in cfg:
                for field in cfg["token_seq_fields"]:
                    lookup_df[table_name][field] = [
                        np.array(list(filter(None, seq.split(" "))))
                        for seq in lookup_df[table_name][field].values
                    ]
        return lookup_df

    def drop_duplicates(self):
        self.lookup_df[self.inter_table] = \
            self.lookup_df[self.inter_table] \
                .drop_duplicates(subset=[self.uid_field, self.item_seq_field])
        self.lookup_df[self.item_table] = \
            self.lookup_df[self.item_table] \
                .drop_duplicates(subset=[self.iid_field])

    def _filter_item_seq_by_num(self, _min, _max):
        assert _min > 0, "min_item_seq_length must be greater than 0"
        if _min is not None and _max is not None:
            _max = float("inf") if _max is None else _max
            _min = 0 if _min is None else _min
            self.lookup_df[self.inter_table] = \
                self.lookup_df[self.inter_table][
                    self.lookup_df[self.inter_table][self.item_seq_field] \
                        .apply(lambda x: len(x) >= _min and len(x) <= _max)
                    ]

    def _map_item_ID(self):
        item_tokens = [self.lookup_df[self.item_table][self.iid_field].values]
        item_tokens.append(
            self.lookup_df[self.inter_table][self.item_seq_field] \
                .agg(np.concatenate)
            )
        split_point = np.cumsum(list(map(len, item_tokens)))[:-1]
        item_tokens = np.concatenate(item_tokens)

        new_ids_list, mappings = pd.factorize(item_tokens)
        [item_tab_new_ids, inter_tab_new_ids] = \
            np.split(new_ids_list + 1, split_point)
        item_id_token = np.array(["[PAD]"] + list(mappings))
        item_token_id = {token: idx for idx, token in enumerate(item_id_token)}

        self.lookup_df[self.item_table][self.iid_field] = item_tab_new_ids
        split_point = np.cumsum(self.lookup_df[self.inter_table][
            self.item_seq_field].agg(len))[:-1]
        self.lookup_df[self.inter_table][self.item_seq_field] = \
            np.split(inter_tab_new_ids, split_point)

        # item already sorted by id when performing factorize
        item_id_text = self.lookup_df[self.item_table][
            self.item_text_field].values
        item_id_text = ["[PAD]"] + item_id_text.tolist()
        return item_token_id, item_id_token, item_id_text
    
    def  _map_user_ID(self):
        user_tokens = [self.lookup_df[self.inter_table][self.uid_field].values]
        user_tokens = np.concatenate(user_tokens)
        new_ids_list, mappings = pd.factorize(user_tokens)
        user_id_token = np.array(["[PAD]"] + list(mappings))
        user_token_id = {token: idx for idx, token in enumerate(user_id_token)}
        self.lookup_df[self.inter_table][self.uid_field] = new_ids_list + 1
        return user_token_id, user_id_token
        
    def save_inters(self, save_dir, suffix=""):
        if self.inter_table in self.processed_df:
            file_name = f"{self.inter_table}_{suffix}.processed.tsv"
            file_path = os.path.join(save_dir, file_name)
            self.processed_df[self.inter_table].to_csv(
                file_path,
                sep="\t",
                index=False,
                encoding="utf-8")

    def save_items(self, save_dir, tokenizer_abbr):
        if self.item_table in self.processed_df:
            file_name = f"{self.item_table}_{tokenizer_abbr}.processed.tsv"
            file_path = os.path.join(save_dir, file_name)
            self.processed_df[self.item_table].to_csv(
                file_path,
                sep="\t",
                index=False,
                encoding="utf-8")
