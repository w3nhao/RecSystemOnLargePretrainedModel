# This script only used for manually pre-inference, after running the pre-inference script.

import sys
import os
# add the realpath of parent directory to the path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))
# delete the current directory from the path
sys.path.remove(current_dir)


import argparse
from datamodules.utils import gather_inference_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="path to the processed directory",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        required=True,
        help="number of items to be pre-inferenced",
    )
    args = parser.parse_args()
    gather_inference_results(args.processed_dir, args.num_items)