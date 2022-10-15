from typing import Optional


def get_data_configs(dataset):
    uid_field = "user_id"
    iid_field = "item_id"
    item_text_field = "item_text"
    item_seq_field = "interactions"

    data_dir = f"data/{dataset}/"

    if dataset in ["MIND_small", "MIND_large"]:
        inter_table = "behaviors"
        item_table = "news"

        old_uid_field = "userid"
        old_iid_field = "newsid"
        old_item_text_field = "title"
        old_item_seq_field = "behaviors"
    elif dataset == "hm":
        inter_table = "behaviors"
        item_table = "items"

        old_uid_field = "userid"
        old_iid_field = "itemid"
        old_item_text_field = "description"
        old_item_seq_field = "behaviors"
    elif dataset == "bilibili":
        inter_table = "behaviors"
        item_table = "videos"

        old_uid_field = "userid"
        old_iid_field = "videoid"
        old_item_text_field = "description"
        old_item_seq_field = "behaviors"
    else:
        raise ValueError(
            "dataset must be in ['MIND_small', 'MIND_large', 'hm', 'bilibili']"
        )

    table_configs = {
        inter_table: {
            "filepath": data_dir + f"{inter_table}.tsv",
            "usecols": [old_uid_field, old_item_seq_field],
            "rename_cols": {
                old_uid_field: uid_field,
                old_item_seq_field: item_seq_field
            },
            "filed_type": {
                old_uid_field: str,
                old_item_seq_field: str
            },
            "token_seq_fields": [item_seq_field],
        },
        item_table: {
            "filepath": data_dir + f"{item_table}.tsv",
            "usecols": [old_iid_field, old_item_text_field],
            "filed_type": {
                "newsid": str,
                "title": str
            },
            "rename_cols": {
                old_iid_field: iid_field,
                old_item_text_field: item_text_field
            },
        },
    }

    data_configs = {
        "data_dir": data_dir,
        "uid_field": uid_field,
        "iid_field": iid_field,
        "item_text_field": item_text_field,
        "item_seq_field": item_seq_field,
        "inter_table": inter_table,
        "item_table": item_table,
    }

    data_configs.update({"table_configs": table_configs})

    return data_configs


class SeqRecDataModuleConfig:

    def __init__(self, dataset: str, **kwargs):
        self.dataset = dataset
        self.plm_name: str = kwargs.pop("plm_name", "facebook/opt-125m")
        self.plm_n_unfreeze_layers: int = \
            kwargs.pop("plm_n_unfreeze_layers", 0)
        self.min_item_seq_len: int = kwargs.pop("min_item_seq_len", 5)
        self.max_item_seq_len: Optional[int] = \
            kwargs.pop("max_item_seq_len", None)
        self.sasrec_seq_len: int = kwargs.pop("sasrec_seq_len", 20)
        self.tokenized_len: int = kwargs.pop("tokenized_len", 30)
        self.batch_size: int = kwargs.pop("batch_size", 64)
        self.num_workers: int = kwargs.pop("num_workers", 4)
        self.pin_memory: bool = kwargs.pop("pin_memory", True)

        assert self.min_item_seq_len > 0
        assert self.tokenized_len > 0
        assert self.sasrec_seq_len > 0
        if self.max_item_seq_len is not None:
            assert self.max_item_seq_len > 0
            assert self.max_item_seq_len >= self.min_item_seq_len

        if self.plm_n_unfreeze_layers is not None:
            assert self.plm_n_unfreeze_layers >= -1
