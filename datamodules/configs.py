def get_configs(data_name):
    uid_field = "user_id"
    iid_field = "item_id"
    item_text_field = "item_text"
    item_seq_field = "interactions"

    if data_name in ["MIND_small", "MIND_large"]:
        data_dir = f"data/{data_name}/"
        inter_table = "behaviors"
        item_table = "news"

        old_uid_field = "userid"
        old_iid_field = "newid"
        old_item_text_field = "title"
        old_item_seq_field = "behaviors"

    else:
        raise ValueError("data_name must be in ['MIND_small', 'MIND_large']")

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
            "usecols": ["newid", "title"],
            "filed_type": {
                "newid": str,
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
