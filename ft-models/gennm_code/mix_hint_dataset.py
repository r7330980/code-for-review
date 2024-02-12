import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
import datasets
from tqdm import tqdm
import json
import os
from datasets import Features
import glob


print()

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default='dirty-mysplit-vannila')
parser.add_argument('--ds_hint', type=str, default='dirty-train-hints')
parser.add_argument('--ds_out', type=str, default='dirty-mix')
parser.add_argument('--concat', action='store_true')

args = parser.parse_args()

ds_hint = datasets.load_dataset(args.ds_hint)
ds_name = datasets.load_dataset(args.ds_name)

if args.concat:
    # simply concat the two datasets, and shuffle
    concated_dataset = datasets.concatenate_datasets([ds_hint['train'], ds_name['train']])
    ret = concated_dataset.shuffle(seed=42)
    new_ds = DatasetDict({'train': ret})
    new_ds.push_to_hub(args.ds_out, private=True)
    exit()

prog_func2entry = {}
for entry in tqdm(ds_name['train'], desc='Collecting name entries'):
    prog_name = os.path.basename(entry['prog_name'])
    prog_func2entry[(prog_name, entry['strip_func_name'])] = entry

ret_entries = []
for entry in tqdm(ds_hint['train'], desc='Collecting hint entries'):
    ret_entries.append(entry)
    if (entry['prog_name'], entry['strip_func_name']) in prog_func2entry:
        ret_entries.append(prog_func2entry[(entry['prog_name'], entry['strip_func_name'])])

more_sample_num = 2 * len(ds_hint['train'])
# random sample more
ds_name = ds_name.shuffle(seed=42)
cnt = 0
pbar = tqdm(total=more_sample_num, desc='Randomly sample more')
for i in range(len(ds_name['train'])):
    entry = ds_name['train'][i]
    if (entry['prog_name'], entry['strip_func_name']) not in prog_func2entry:
        cnt += 1
        pbar.update(1)
        ret_entries.append(ds_name['train'][i])
    if cnt >= more_sample_num:
        break

new_ds = Dataset.from_list(ret_entries)
new_ds = new_ds.shuffle(seed=42)
new_ds = DatasetDict({'train': new_ds})
    

# push to hub
new_ds.push_to_hub(args.ds_out, private=True)

print()
