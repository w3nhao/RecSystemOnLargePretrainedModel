# This script only used for manually pre-inference, after running the pre-inference script.

import sys
import os
# add the realpath of parent directory to the path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))
# delete the current directory from the path
sys.path.remove(current_dir)


import argparse
from datamodules.datamodule import PreInferSeqDataModule


_CUDAS = [0, 1, 2, 3, 4, 5, 6, 7]


def int_or_none(x):
    return None if x.lower() == "none" else int(x)


if __name__ == "__main__":
    
    
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--dataset", type=str, default="MIND_small")
    argparser.add_argument("--plm_name", type=str, default="facebook/opt-125m")
    argparser.add_argument("--plm_last_n_unfreeze", type=int, default=0)
    argparser.add_argument("--sasrec_seq_len", type=int, default=20)
    argparser.add_argument("--tokenized_len", type=int, default=30)
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--num_workers", type=int, default=4)
    argparser.add_argument("--pin_memory", type=bool, default=False)
    argparser.add_argument("--min_item_seq_len", type=int, default=5)
    argparser.add_argument("--max_item_seq_len", type=int_or_none, default=None)
    argparser.add_argument("--pre_inference_devices", type=int, nargs="+", default=_CUDAS)
    argparser.add_argument("--pre_inference_precision", type=int, default=32)
    argparser.add_argument("--pre_inference_batch_size", type=int, default=1)
    argparser.add_argument("--pre_inference_num_workers", type=int, default=4)
    args, _ = argparser.parse_known_args()
    
    dm_config = PreInferSeqDataModule.build_datamodule_config(args=args)
    dm = PreInferSeqDataModule(dm_config)