import os
import numpy as np
import pandas as pd
from utils.pylogger import get_pylogger

from transformers import GPT2Tokenizer

ITEM_ID_SEQ_FIELD = "input_seqs"
TARGET_FIELD = "targets"
TEXT_ID_SEQ_FIELD = "tokenized_ids"
ATTENTION_MASK_FIELD = "attention_mask"

log = get_pylogger(__name__)


def str_fields2ndarray(df, fields, field_len, dtype=np.int64):
    """Convert string fields to np array"""
    features = {}
    for field in fields:
        features[field] = np.empty((len(df), field_len), dtype=dtype)
        for i, seq in enumerate(df[field].values):
            seq_arr = np.array(list(seq.split(" ")), dtype=dtype)
            features[field][i] = seq_arr
    ndarrays = [features[f] for f in fields]
    return ndarrays


def right_padding_left_trancate(data, seq_len):
    """Generate items seq like [1, 2, 3, 0 , 0] and targets like [2, 3, 4, 0, 0]"""
    item_seqs = np.zeros((len(data), seq_len), dtype=np.int64)
    targets = np.zeros((len(data), seq_len), dtype=np.int64)
    for i, data in enumerate(data):
        if len(data) > seq_len:
            item_seqs[i] = data[-seq_len - 1:-1]
            targets[i] = data[-seq_len:]
        else:
            item_seqs[i, :len(data) - 1] = data[:-1]
            targets[i, :len(data) - 1] = data[1:]

    return item_seqs, targets


def ratio_split(data, ratios):
    """Split data into train, valid and test set by ratio"""
    """ fixed ratio split (train:0.8, valid:0.1, test:0.1) """
    assert sum(ratios) == 1.0
    train_ratio, valid_ratio, test_ratio = ratios
    train_data = data.sample(frac=train_ratio)
    rest_data = data[~data.index.isin(train_data.index)]
    valid_data = rest_data.sample(frac=valid_ratio /
                                  (valid_ratio + test_ratio))
    test_data = rest_data[~rest_data.index.isin(valid_data.index)]

    return train_data, valid_data, test_data


def tokenize(text_seqs, tokenizer, tokenized_len):
    """Tokenize text seqs"""
    tokenized_seqs = tokenizer(
        text_seqs,
        return_tensors="pt",
        max_length=tokenized_len,
        padding="max_length",
        truncation="longest_first",
    )
    tokenized_seqs = {
        TEXT_ID_SEQ_FIELD: tokenized_seqs["input_ids"].numpy(),
        ATTENTION_MASK_FIELD: tokenized_seqs["attention_mask"].numpy(),
    }
    return tokenized_seqs


class DataPreporcessor:

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

        self.item_token_id, self.item_id_token, self.item_id_text = None, None, None

        self.lookup_df = self._load_data()
        self.processed_df = {}

    @property
    def num_items(self):
        return len(self.item_id_token)

    @property
    def min_item_seq_len(self):
        return self.lookup_df[self.inter_table][self.item_seq_field].apply(
            len).min()

    @property
    def max_item_seq_len(self):
        return self.lookup_df[self.inter_table][self.item_seq_field].apply(
            len).max()

    def prepare_inters(self, sasrec_seq_len):
        """Prepare interactions data"""
        inters = self.lookup_df[self.inter_table][self.item_seq_field].values
        item_seqs, targets = right_padding_left_trancate(
            inters, sasrec_seq_len)

        item_seqs = np.array([" ".join(seq) for seq in item_seqs.astype(str)])
        targets = np.array([" ".join(seq) for seq in targets.astype(str)])

        self.processed_df[self.inter_table] = pd.DataFrame({
            "input_seqs": item_seqs,
            "targets": targets
        })

    def prepare_items(self, pretrained_model, tokenized_len):
        """Prepare items data"""
        if pretrained_model.startswith("facebook/opt"):
            tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        else:
            raise NotImplementedError

        tokenized_seqs = tokenize(self.item_id_text, tokenizer, tokenized_len)

        tokenized_ids = tokenized_seqs[TEXT_ID_SEQ_FIELD].astype(np.str_)
        attention_mask = tokenized_seqs[ATTENTION_MASK_FIELD].astype(np.str_)

        tokenized_ids = np.array([" ".join(seq) for seq in tokenized_ids])
        attention_mask = np.array([" ".join(seq) for seq in attention_mask])

        self.processed_df[self.item_table] = pd.DataFrame({
            TEXT_ID_SEQ_FIELD:
            tokenized_ids,
            ATTENTION_MASK_FIELD:
            attention_mask,
        })

    def prepare_data(self):
        # self.drop_duplicates()
        self.item_token_id, self.item_id_token, self.item_id_text = self._map_item_ID(
        )
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
        self.lookup_df[self.inter_table] = self.lookup_df[
            self.inter_table].drop_duplicates(
                subset=[self.uid_field, self.item_seq_field])
        self.lookup_df[self.item_table] = self.lookup_df[
            self.item_table].drop_duplicates(subset=[self.iid_field])

    def _filter_item_seq_by_num(self, _min, _max):
        assert _min > 0, "min_item_seq_length must be greater than 0"
        if _min is not None and _max is not None:
            _max = float("inf") if _max is None else _max
            _min = 0 if _min is None else _min
            self.lookup_df[self.inter_table] = self.lookup_df[
                self.inter_table][self.lookup_df[self.inter_table][
                    self.item_seq_field].apply(
                        lambda x: len(x) >= _min and len(x) <= _max)]

    def _map_item_ID(self):
        item_tokens = [self.lookup_df[self.item_table][self.iid_field].values]
        item_tokens.append(
            self.lookup_df[self.inter_table][self.item_seq_field].agg(
                np.concatenate))
        split_point = np.cumsum(list(map(len, item_tokens)))[:-1]
        item_tokens = np.concatenate(item_tokens)

        new_ids_list, mappings = pd.factorize(item_tokens)
        [item_tab_new_ids,
         inter_tab_new_ids] = np.split(new_ids_list + 1, split_point)
        item_id_token = np.array(["[PAD]"] + list(mappings))
        item_token_id = {token: idx for idx, token in enumerate(item_id_token)}

        self.lookup_df[self.item_table][self.iid_field] = item_tab_new_ids
        split_point = np.cumsum(self.lookup_df[self.inter_table][
            self.item_seq_field].agg(len))[:-1]
        self.lookup_df[self.inter_table][self.item_seq_field] = np.split(
            inter_tab_new_ids, split_point)

        # item already sorted by id when performing factorize
        item_id_text = self.lookup_df[self.item_table][
            self.item_text_field].values
        item_id_text = ["[PAD]"] + item_id_text.tolist()
        return item_token_id, item_id_token, item_id_text

    def save_data(self, save_dir):
        for table_name, df in self.processed_df.items():
            df.to_csv(
                os.path.join(save_dir, table_name + ".processed.tsv"),
                sep="\t",
                index=False,
                encoding="utf-8",
            )
