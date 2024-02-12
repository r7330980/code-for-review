import json
from tqdm import tqdm
import utils
import eval_utils
from matplotlib import pyplot as plt
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ret-in', type=str, default='')
parser.add_argument('--fout', type=str, default='')
args = parser.parse_args()

ret_fin = open(args.ret_in, 'r')

data_raw = [json.loads(line) for line in ret_fin.readlines()]
ret_fin.close()
ret_out = []
for entry in tqdm(data_raw):
    prog_name = entry['prog_name']
    func_name = entry['func_name']
    ori_varname = entry['varname']
    pred_name = entry['pred_name']
    gt_varname = entry['gt_varname']
    pr, rc = eval_utils.score_name_ori(gt_varname, pred_name)
    entry['precision'] = pr
    entry['recall'] = rc
    ret_out.append(entry)

if args.fout == "":
    args.fout = args.ret_in + ".converted.jsonl"

with open(args.fout, 'w') as fout:
    for entry in ret_out:
        fout.write(json.dumps(entry) + "\n")
print(f"converted to {args.fout}")