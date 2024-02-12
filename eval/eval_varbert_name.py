import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import utils
import eval_utils
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fin', type=str, default='')
parser.add_argument('--fout', type=str, default='')
parser.add_argument('--ori', action='store_true')

args = parser.parse_args()

fin = open(args.fin, 'r').readlines()
data_in = [json.loads(l.strip()) for l in fin]

ret = []
for entry in tqdm(data_in, desc="evaluating"):
    gt_varname = entry['gt_varname']
    pred_name = entry['pred_name']
    if 'UNK' == pred_name:
        pr = 0
        rc = 0
        pred_name = "<emptyname>"
    else:
        if args.ori:
            pr, rc = eval_utils.score_name_ori(gt_varname, pred_name)
        else:
            pr, rc = eval_utils.score_name(gt_varname, pred_name)
    ret.append({
        **entry,
        'precision': pr,
        'recall': rc,
    })


if args.fout == "":
    if args.ori:
        args.fout = args.fin + ".evaluated.ori.jsonl"
    else:
        args.fout = args.fin + ".evaluated.jsonl"
with open(args.fout, 'w') as fout:
    for entry in ret:
        fout.write(json.dumps(entry) + "\n")


    
