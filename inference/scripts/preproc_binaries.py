import sys
sys.path.append("model")
from transformers import (
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    LlamaForCausalLM,
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
import gzip
import os
import transformers
import torch
import json
from tqdm import tqdm
import numpy as np
import name_utils
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx
import pickle
import multiprocessing
import functools
from binary_prog import BinaryProgram


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
transformers.utils.logging.set_verbosity_info()
log_level = logging.INFO
logger.setLevel(log_level)

transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


@dataclass
class GenNmPreprocArguments:
    prog_list: Optional[str] = field(
        default="test_prog.txt",
    )

    fout: Optional[str] = field(
        default="test_prog.preproc.pkl",
    )

    num_workers: Optional[int] = field(
        default=1,
    )

    num_splits: Optional[int] = field(
        default=20,
    )



            
def load_one_binary(fname, idx):
    global pbar
    if idx % 10 == 0:     
      # update pbar to idx
      step = idx - pbar.n
      pbar.update(step)
    if not os.path.exists(fname):
        return None
    current_fin_lines = gzip.open(fname, 'r').readlines()
    current_binary_funcs = []
    for line in current_fin_lines:
        str_line = line.decode('utf-8', errors='ignore')
        current_binary_funcs.append(json.loads(str_line))
    prog_name = os.path.basename(fname)
    bin_prog = BinaryProgram(prog_name=prog_name,
                        func_entry_list=current_binary_funcs)
    return bin_prog
    
def main():
    parser = HfArgumentParser((GenNmPreprocArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        gennm_args = parser.parse_yaml_file(
            json_file=os.path.abspath(sys.argv[1])[0]
        )
    else:
        gennm_args = parser.parse_args_into_dataclasses()[0]

    prog_list = open(gennm_args.prog_list).readlines()
    prog_list = [prog.strip() for prog in prog_list]

    global pbar

    pbar = tqdm(total=len(prog_list))
    # load_one_binary_partial = functools.partial(load_one_binary, pbar=pbar)


    # load all binaries with multiprocessing
    with multiprocessing.Pool(gennm_args.num_workers) as p:
        bin_progs = p.starmap(load_one_binary, [(fname, idx) for idx, fname in enumerate(prog_list)])
    bin_progs = [prog for prog in bin_progs if prog is not None]
    print("Loaded %d binaries"%len(bin_progs))
    print("Dividing into %d splits"%gennm_args.num_splits)
    all_progs = len(bin_progs)    
    for i in tqdm(range(gennm_args.num_splits)):
        start = int(i * all_progs / gennm_args.num_splits)
        end = int((i+1) * all_progs / gennm_args.num_splits)
        pickle.dump(bin_progs[start:end], open(gennm_args.fout + ".%d" % i, 'wb'))


if __name__ == "__main__":
    main()
