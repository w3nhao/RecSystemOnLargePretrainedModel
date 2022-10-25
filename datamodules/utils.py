import os
import torch
import subprocess
import time
import numpy as np
from transformers import AutoConfig
from utils.pylogger import get_pylogger
from models.utils import PRETRAIN_MODEL_ABBR
from tqdm import tqdm

USER_ID_FIELD = "user_id"
ITEM_ID_SEQ_FIELD = "input_seqs"
TARGET_FIELD = "targets"
TEXT_ID_SEQ_FIELD = "tokenized_ids"
ATTENTION_MASK_FIELD = "attention_mask"

GLOBAL_RANDOM_SEED = 42

log = get_pylogger(__name__)


def neg_sampling(batch, n_samping, n_items):
    neg_samples = torch.randint(0, n_items, (n_samping, *batch.shape))
    # mark which sample is positive
    pos_mask = (neg_samples == batch.unsqueeze(0))
    while pos_mask.any():
        neg_samples[pos_mask] = torch.randint(0, n_items, (pos_mask.sum(),))
        pos_mask = (neg_samples == batch.unsqueeze(0))
    return batch, neg_samples


def str_fields2ndarray(df, fields, field_lens, dtype=np.int64):
    """Convert string fields in pandas df to np array"""
    """ '1 2 3' -> [1, 2, 3] """
    assert isinstance(fields, list)
    assert isinstance(field_lens, list)
    assert len(fields) == len(field_lens)
    
    features = {}
    for field, field_len in zip(fields, field_lens):
        if field_len is None:
            log.warning(f"field_len is None, the ndarray of field {field} will be in "
                        f"variable length and dtype=object")
            features[field] = np.empty((len(df), ), dtype=object)
        else:
            features[field] = np.empty((len(df), field_len), dtype=dtype)
        for i, seq in enumerate(df[field].values):
            seq_arr = np.array(list(seq.split(" ")), dtype=dtype)
            features[field][i] = seq_arr
            
    if len(fields) == 1:
        ndarrays = features[fields[0]]
    else:
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

def seq_leave_one_out_split(input_seqs, targets, pad_id=0):
    """Leave one out split"""
    """ split data into train , val and test """
    input_seqs_dict = {}
    targets_dict = {}
    pad_id=0
    masks = np.where(input_seqs != pad_id, True, False)
    last_idx = np.sum(masks, axis=1) - 1
    last_idx = np.expand_dims(last_idx, axis=1)
    for stage in ["test", "val", "train"]:
        input_seqs_dict[stage] = input_seqs.copy()
        targets_dict[stage] = targets.copy()
        if stage != "test":
            np.put_along_axis(input_seqs_dict[stage], last_idx, pad_id, axis=1)
            np.put_along_axis(targets_dict[stage], last_idx, pad_id, axis=1)
            last_idx = np.concatenate([last_idx, last_idx-1], axis=1)
            
    return [
        (input_seqs_dict[stage], targets_dict[stage]) 
        for stage in ["train", "val", "test"]
        ]
    
def point_wise_leave_one_out_split(user_ids, item_id_seqs):
    """ Leave one out split for point wise data """
    """ split data into train , val and test """
    item_ids_dict = {}
    for uid, item_id_seq in zip(user_ids, item_id_seqs):
        item_ids_dict["train"] = item_id_seq[:-2]
        item_ids_dict["val"] = item_id_seq[-2:-1]
        item_ids_dict["test"] = item_id_seq[-1:]
    return [
        (user_ids, item_ids_dict[stage]) 
        for stage in ["train", "val", "test"]
        ]

def ratio_split(data, ratios, random_state):
    """Split data into train, valid and test set by ratio"""
    """ fixed ratio split (train:0.8, valid:0.1, test:0.1) """
    assert sum(ratios) == 1.0
    train_ratio, valid_ratio, test_ratio = ratios
    train_data = data.sample(
        frac=train_ratio, random_state=random_state)
    rest_data = data[~data.index.isin(train_data.index)]
    valid_frac = valid_ratio / (valid_ratio + test_ratio)
    valid_data = rest_data.sample(
        frac=valid_frac, random_state=random_state)
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


def call_inference(
    processed_dir,
    item_file,
    plm_name,
    input_layer="none",
    target_layer="emb",
    input_item_embs_file=None,
    tokenized_len=30,
    batch_size=1,
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
    precision=32,
    num_workers=4,
    inference_script_path="scripts/inference.py",
):
    """Call inference script to infer item embs"""

    if input_item_embs_file is None:
        input_item_embs_file = "none"

    cmd = ["python", inference_script_path]
    cmd += ["--plm_name", plm_name]
    cmd += ["--processed_dir", processed_dir]
    cmd += ["--processed_items_file", item_file]
    cmd += ["--input_item_embs_file", input_item_embs_file]
    cmd += ["--input_layer", str(input_layer)]
    cmd += ["--target_layer", str(target_layer)]
    cmd += ["--tokenized_len", str(tokenized_len)]
    cmd += ["--batch_size", str(batch_size)]
    cmd += ["--devices"] + [str(d) for d in devices]
    cmd += ["--precision", str(precision)]
    cmd += ["--num_workers", str(num_workers)]
    code = subprocess.call(cmd)
    return code


def pre_inference(
    plm_name,
    last_n_unfreeze,
    processed_dir,
    item_file,
    tokenized_len=30,
    batch_size=1,
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
    precision=32,
    num_workers=4,
    inference_script_path="scripts/inference.py",
    layer_wise=True,
):
    """Pre-inference for the last n layers of the plm"""

    plm_config = AutoConfig.from_pretrained(plm_name)
    file_processor = InferenceFileProcessor(
            processed_dir=processed_dir,
            plm_name=plm_name,
            last_n_unfreeze=last_n_unfreeze,
            n_inferenc_devices=len(devices),
        )

    if file_processor.exists_inference_file():
        embs_file = file_processor.get_inference_file()
        log.info(
            f"already inferenced {plm_name} for unfreeze last {last_n_unfreeze} layers"
        )
        return torch.load(os.path.join(processed_dir, embs_file))

    log.info(
        f"start inferencing {plm_name} for unfreeze last {last_n_unfreeze} layers..."
    )
    
    final_n_freeze = plm_config.num_hidden_layers - last_n_unfreeze
    
    levels = ["none", "emb"] + [i for i in range(final_n_freeze)]
    prev_embs_file = file_processor.find_nearest_inference_file()
    if prev_embs_file is not None:
        prev_n_unfreeze, prev_embs_file = prev_embs_file
        input_layer = levels[prev_n_unfreeze + 1]
    else:
        input_layer = "none"
        prev_embs_file = "none"
        
    # layer-wise inference
    if layer_wise:
        # take OPT125M as an example: 
        # levels = ["none", "emb", 0, 1, 2, 3, 4 ... 11]
        # if previous target layer as input layer is 2, which means we have inferenced 
        # to the 2nd decoder layer, so now we have a file prefix as "OPT125M_freeze@2", 
        # n_freeze will be 3 after this call, and the current target layer will be 
        # levels[3 + 1] = 2 (levels[n_freeze + 1] = 2), which is the 3rd decoder layer
        start = levels.index(input_layer)
        end = levels.index(final_n_freeze - 1) if final_n_freeze != 0 else len(levels) - 1
        for n_freeze in range(start, end):
            prev_layer = levels[n_freeze]
            curr_layer = levels[n_freeze + 1]
            return_code = call_inference(
                processed_dir=processed_dir,
                item_file=item_file,
                plm_name=plm_name,
                input_layer=prev_layer,
                target_layer=curr_layer,
                input_item_embs_file=prev_embs_file,
                tokenized_len=tokenized_len,
                batch_size=batch_size,
                devices=devices,
                precision=precision,
                num_workers=num_workers,
                inference_script_path=inference_script_path,
            )
            if return_code == 0:
                new_embs_file = file_processor.gather_inference_results(
                    n_freeze=n_freeze)
                # only keep inferenced files for the last 2, 1, 0 unfreeze models
                if plm_config.num_hidden_layers - n_freeze >= 2 and prev_embs_file != "none":
                    os.remove(os.path.join(processed_dir, prev_embs_file))
                prev_embs_file = new_embs_file
            else:
                raise RuntimeError("Pre-inference failed.")
    # full inference
    else:
        target_layer = levels[-1]
        return_code = call_inference(
            processed_dir=processed_dir,
            item_file=item_file,
            plm_name=plm_name,
            input_layer=input_layer,
            target_layer=target_layer,
            input_item_embs_file=prev_embs_file,
            tokenized_len=tokenized_len,
            batch_size=batch_size,
            devices=devices,
            precision=precision,
            num_workers=num_workers,
            inference_script_path=inference_script_path,
        )
        if return_code == 0:
            new_embs_file = file_processor.gather_inference_results(
                n_freeze=final_n_freeze)
        else:
            raise RuntimeError("Pre-inference failed.")
        
        log.info(
            f"finish inferencing {plm_name} for unfreeze last {last_n_unfreeze} layers"
        )

class InferenceFileProcessor:

    def __init__(self,
                 processed_dir,
                 plm_name,
                 last_n_unfreeze,
                 n_inferenc_devices=None):
        self.plm_name = plm_name
        self.plm_abbr = PRETRAIN_MODEL_ABBR[plm_name]
        self.plm_config = AutoConfig.from_pretrained(plm_name)
        assert self.plm_config.num_hidden_layers >= last_n_unfreeze
        assert last_n_unfreeze > -1
        if last_n_unfreeze == self.plm_config.num_hidden_layers:
            log.info("last_n_unfreeze is equal to the number of layers, only freeze the embedding layer")
        self.last_n_unfreeze = last_n_unfreeze
        self.n_freeze = self.plm_config.num_hidden_layers - last_n_unfreeze
        self.processed_dir = processed_dir
        self.n_inferenc_devices = n_inferenc_devices

    def get_inference_file(self, is_embs=True, rank_idx=None):
        """
        Get the inference file name
        
        Distributed inference will generate multiple files, so we need to specify the rank_idx
        
        Format: 
            {plm_abbr}_freeze@{n_freeze}_inferenced_{embs/idxs}_rank@{rank_idx}.pt
        Example: 
            OPT125M_freeze@10_inferenced_embs_for_unfreeze@2_rank\@0.pt
            
        If rank_idx is None, then it will return the file name for the final inference result
        
        Format: 
            {plm_abbr}_freeze@{n_freeze}_inferenced_embs.pt
        Example: 
            OPT125M_freeze@10_inferenced_embs.pt
        """
        file_type = "embs" if is_embs else "idxs"
        _file_name = f"{self.plm_abbr}_" + \
                    f"freeze@{self.n_freeze}_" + \
                    f"inferenced_{file_type}"
        if rank_idx is not None:
            file_name = _file_name + f"_rank@{rank_idx}.pt"
        else:
            file_name = _file_name + ".pt"
        return file_name

    def get_files_in_processed_dir(self):
        """ Get all files in the processed_dir """
        files = []
        for f in os.listdir(self.processed_dir):
            if os.path.isfile(os.path.join(self.processed_dir, f)):
                files.append(f)
        return files

    def exists_inference_file(self):
        """ Check if the inference file exists """
        # get all files in the processed_dir
        files = self.get_files_in_processed_dir()

        already_inferenced = False
        # inferenced result name format:
        # OPT125M_freeze@10_inferenced_embs.pt
        for f in files:
            if f == self.get_inference_file():
                already_inferenced = True
                break
        return already_inferenced

    def find_nearest_inference_file(self):
        """ Get the nearest inference file """
        if self.exists_inference_file():
            return self.get_inference_file()

        files = self.get_files_in_processed_dir()

        emb_files, n_freeze_list = [], []
        file_prefix = f"{self.plm_abbr}_freeze@"
        for f in files:
            if f.startswith(file_prefix) and f.endswith("_inferenced_embs.pt"):
                n_freeze = int(f.split("@")[1].split("_")[0])
                n_freeze_list.append(n_freeze)
                emb_files.append(f)

        if len(emb_files) == 0:
            log.info(f"no inference file found in {self.processed_dir}")
            return None

        difference = [self.n_freeze - n for n in n_freeze_list]
        lower_layer_diff = [diff for diff in difference if diff > 0]
        min_diff = min(lower_layer_diff) if len(lower_layer_diff) > 0 else None
        if min_diff is None:
            log.info(f"no lower layer inference file found in {self.processed_dir}")
            return None
        else:
            nearest_n_freeze = self.n_freeze - min_diff
            nearest_n_freeze_idx = difference.index(min_diff)
            nearest_file = emb_files[nearest_n_freeze_idx]
            return nearest_n_freeze, nearest_file

    def gather_inference_results(self, n_freeze):
        """collect inference results from all devices"""
        # inferenced result name format:
        # OPT125M_freeze@10_inferenced_idxs_rank@0.pt
        # OPT125M_freeze@10_inferenced_embs_rank@0.pt
        # the last number before .pt is the device id
        file_prefix = f"{self.plm_abbr}_freeze@{n_freeze}_inferenced"
        
        max_rank = self.n_inferenc_devices - 1

        retry_times = 10
        for i in range(retry_times):  
            inferenced_embs, inferenced_idxs = [], []
            try:      
                for i in range(max_rank + 1):
                    idxs_file = f"{file_prefix}_idxs_rank@{i}.pt"
                    embs_file = f"{file_prefix}_embs_rank@{i}.pt"
                    idxs = torch.load(os.path.join(self.processed_dir, idxs_file))
                    embs = torch.load(os.path.join(self.processed_dir, embs_file))
                    inferenced_idxs.append(idxs)
                    inferenced_embs.append(embs)
            except Exception as e:
                n_times = i + 1
                log.error(f"The {n_times} time fail to gather inference results, "
                          f"remaining {retry_times - n_times} to try, error: \n {e}")
                
                for _ in tqdm(range(60), desc=f"waiting for inference results"):
                    time.sleep(1)
                continue
            
        if inferenced_embs == []:
            raise RuntimeError(
                f"Failed to gather inference results after {retry_times} times retry.")

        # concat all inferenced results and sort by idxs
        inferenced_idxs = torch.cat(inferenced_idxs, dim=0)
        sorted_idxs = torch.argsort(inferenced_idxs)

        inferenced_embs = torch.cat(inferenced_embs, dim=0)
        sorted_embs = inferenced_embs[sorted_idxs]
        
        files = self.get_files_in_processed_dir()
        # remove the tmp result files
        for f in files:
            if file_prefix in f and "rank" in f:
                os.remove(os.path.join(self.processed_dir, f))

        result_file = f"{file_prefix}_embs.pt"
        torch.save(sorted_embs, os.path.join(self.processed_dir, result_file))

        # return the final result file name
        return result_file

