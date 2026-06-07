#!/usr/bin/env python
#  coding=utf-8

"""
Samples an existing RNN model.
"""

import argparse
import gzip
import functools

import tqdm

import models.model as mm
#import utils.log as ul


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Samples a model.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--output-smiles-path", "-o",
                        help="Path to the output file (if none given it will use stdout).", type=str)
    parser.add_argument("--num", "-n", help="Number of SMILES to sample [DEFAULT: 1024]", type=int, default=1024)
    parser.add_argument("--with-nll", help="Store the NLL in a column after the SMILES.",
                        action="store_true", default=False)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size (beware GPU memory usage) [DEFAULT: 128]", type=int, default=128)
    parser.add_argument("--use-gzip", help="Compress the output file (if set).", action="store_true", default=False)

    return parser.parse_args()


def main():
    """Main function."""
    #args = parse_args()
    import sys
    import os
    
    #task_name=sys.argv[1]
    #model_path="/home/zhangky/tools/reinvent-randomized/"+task_name+"/models/" #model.trained.57#
    model_path="/home/zhangky/tool/swit/gen_models/data/"  # "/home/zhangky/tool/swit/best_gm/"
    model_name="augmented.prior"
    model = mm.Model.load_from_file(model_path+model_name)
    print(model.network) #.state_dict
    import torch.nn as nn

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    num_parameters = count_parameters(model.network)
    print(model_path+model_name)
    print(f'num_parameters: {num_parameters}')


#LOG = ul.get_logger(name="sample_from_model")
if __name__ == "__main__":
    main()
