import json
from tqdm import tqdm
import utils
import eval_utils
from matplotlib import pyplot as plt
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--major', type=str, default='key-rets/dirty-valid-cls-retall-18k-rnd0.jsonl.out.jsonl')
parser.add_argument('--default', type=str, default='key-rets/dirty-valid-gen-retall-13k-rnd0.jsonl.out.jsonl')
parser.add_argument('--fout', type=str, default='key-rets/dirty-valid-naive-combine-rnd0.jsonl.out.jsonl')

args = parser.parse_args()

major_names = [json.loads(l.strip()) for l in open(args.major, 'r').readlines()]
default_names = [json.loads(l.strip()) for l in open(args.default, 'r').readlines()]

default_name_map = {}
for entry in default_names:
  prog_name = entry['prog_name']
  func_name = entry['func_name']
  varname = entry['varname']
  if (prog_name, func_name, varname) in default_name_map:
    continue
  default_name_map[(prog_name, func_name, varname)] = entry


combined_names = []
for entry in tqdm(major_names, desc='Combining'):
  prog_name = entry['prog_name']
  func_name = entry['func_name']
  varname = entry['varname']
  pred_name = entry['pred_name']
  if '<emp' in pred_name or '<unk' in pred_name:
    if (prog_name, func_name, varname) in default_name_map:
      combined_names.append(default_name_map[(prog_name, func_name, varname)])
      continue    
  combined_names.append(entry)


fout = open(args.fout, 'w')
for entry in combined_names:
  fout.write(json.dumps(entry) + '\n')
fout.close()

print()

