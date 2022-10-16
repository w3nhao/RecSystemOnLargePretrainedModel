import os
import torch
import subprocess
import numpy as np

ITEM_ID_SEQ_FIELD = "input_seqs"
TARGET_FIELD = "targets"
TEXT_ID_SEQ_FIELD = "tokenized_ids"
ATTENTION_MASK_FIELD = "attention_mask"

PRETRAIN_MODEL_ABBR = {
    "facebook/opt-125m": "OPT125M",
    "facebook/opt-350m": "OPT350M",
    "facebook/opt-1.3b": "OPT1.3B",
    "facebook/opt-2.7b": "OPT2.7B",
    "facebook/opt-6.7b": "OPT6.7B",
    "facebook/opt-13b": "OPT13B",
    "facebook/opt-30b": "OPT30B",
    "facebook/opt-66b": "OPT66B",
    "bert-base-uncased": "BERTBASE",
    "bert-large-uncased": "BERTLARGE",
}


def str_fields2ndarray(df, fields, field_len, dtype=np.int64):
    """Convert string fields in pandas df to np array"""
    """ '1 2 3' -> [1, 2, 3] """
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


def leave_one_out_split(data):
    """Leave one out split"""
    """ split data into train , val and test """
    pass


def ratio_split(data, ratios):
    """Split data into train, valid and test set by ratio"""
    """ fixed ratio split (train:0.8, valid:0.1, test:0.1) """
    assert sum(ratios) == 1.0
    train_ratio, valid_ratio, test_ratio = ratios
    train_data = data.sample(frac=train_ratio)
    rest_data = data[~data.index.isin(train_data.index)]
    valid_frac = valid_ratio / (valid_ratio + test_ratio)
    valid_data = rest_data.sample(frac=valid_frac)
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


def inferenced_embs_exists(processed_dir, plm_name, plm_last_n_unfreeze):
    
    # get all files in the processed_dir
    files = []
    for f in os.listdir(processed_dir):
        if os.path.isfile(os.path.join(processed_dir, f)):
            files.append(f)

    already_inferenced = False
    # inferenced result name format:
    # OPT125M_freeze@10_inferenced_embs_for_unfreeze@2.pt
    embs_file = None
    for f in files:
        if f.endswith(f"unfreeze@{plm_last_n_unfreeze}.pt"):
            if f.startswith(PRETRAIN_MODEL_ABBR[plm_name]):
                embs_file = f
                already_inferenced = True
                return already_inferenced, embs_file
    return already_inferenced, embs_file


def call_pre_inference(
    processed_dir,
    item_file,
    plm_name,
    plm_last_n_unfreeze,
    tokenized_len=30,
    batch_size=1,
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
    precision=32,
    num_workers=4,
    inference_script_path="scripts/preinference.py",
):
    cmd = ["python", inference_script_path]
    cmd += ["--processed_dir", processed_dir]
    cmd += ["--processed_items_file", item_file]
    cmd += ["--plm_name", plm_name]
    cmd += ["--plm_last_n_unfreeze", str(plm_last_n_unfreeze)]
    cmd += ["--tokenized_len", str(tokenized_len)]
    cmd += ["--batch_size", str(batch_size)]
    cmd += ["--devices"] + [str(d) for d in devices]
    cmd += ["--precision", str(precision)]
    cmd += ["--num_workers", str(num_workers)]
    code = subprocess.call(cmd)
    return code


def gather_inference_results(processed_dir, num_items):
    """collect inference results from all devices"""
    
    # get all files in the processed_dir
    files = []
    for f in os.listdir(processed_dir):
        if os.path.isfile(os.path.join(processed_dir, f)):
            files.append(f)

    # inferenced result name format:
    # OPT125M_freeze@10_inferenced_idxs_for_unfreeze@2_0.pt
    # OPT125M_freeze@10_inferenced_embs_for_unfreeze@2_0.pt
    # the last number before .pt is the device id
    plms = []
    for f in files:
        if "inferenced_idxs" in f:
            plm_name = f.split("_")[0]
            n_freeze_layers = f.split("_")[1]
            n_unfreeze_layers = f.split("_")[5]
            plms.append(f"{plm_name}_{n_freeze_layers}+{n_unfreeze_layers}")
    plms = list(set(plms))

    max_ranks = []
    for plm in plms:
        ranks = []
        for f in files:
            if plm.split("+")[0] + "_inferenced_idxs" in f:
                ranks.append(f.split("_")[-1].split(".")[0])
        max_ranks.append(max(int(rank) for rank in ranks))

    sorted_embs = None
    for plm, max_rank in zip(plms, max_ranks):
        inferenced_embs = []
        inferenced_idxs = []
        for i in range(max_rank + 1):
            name_n_freeze = plm.split("+")[0] # OPT125M_freeze@10
            n_unfreeze = plm.split("+")[1] # unfreeze@2
            idxs = torch.load(
                os.path.join(
                    processed_dir, 
                    f"{name_n_freeze}_inferenced_idxs_"
                    f"for_{n_unfreeze}_{i}.pt"
                    )
                )
            embs = torch.load(
                os.path.join(
                    processed_dir, 
                    f"{name_n_freeze}_inferenced_embs_"
                    f"for_{n_unfreeze}_{i}.pt"
                    )
                )
            inferenced_idxs.append(idxs)
            inferenced_embs.append(embs)

        # concat all inferenced results and sort by idxs
        inferenced_idxs = torch.cat(inferenced_idxs, dim=0)
        sorted_idxs = torch.argsort(inferenced_idxs)

        if len(sorted_idxs) != num_items:
            raise ValueError(
                f"num_items: {num_items}, sorted_idxs: {len(sorted_idxs)}"
                "the number of inferenced items is not equal to the number of items"
                "probably some items are oversampled when using the ddp sampler"
                "please use single accelerator to infer the embeddings")

        inferenced_embs = torch.cat(inferenced_embs, dim=0)
        sorted_embs = inferenced_embs[sorted_idxs]

        # remove the files
        for f in files:
            if f"{name_n_freeze}_inferenced_" in f:
                os.remove(os.path.join(processed_dir, f))

        torch.save(sorted_embs,
                   os.path.join(
                        processed_dir, 
                        f"{name_n_freeze}_inferenced_embs_"
                        f"for_{n_unfreeze}.pt"))

    # only return one plm inferenced embs if needed
    return sorted_embs