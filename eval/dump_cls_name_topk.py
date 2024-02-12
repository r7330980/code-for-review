import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
from transformers import (
    CodeLlamaTokenizer, LlamaForCausalLM, set_seed,
    HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, TrainingArguments)
import utils
import eval_utils
from tqdm import tqdm
from hashlib import md5

parser = argparse.ArgumentParser()
parser.add_argument('--fin', type=str, default='')
parser.add_argument('--fout', type=str, default='')
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--force-parse', action='store_true')
parser.add_argument('--case-study', action='store_true')
args = parser.parse_args()
# # # XXX: for debug
# args.force_parse = True

def evaluate_batch(data):
  ret = []
  error_cnt = 0
  total_cnt = 0
  fixed_cnt = 0  
  dbg_error_entries = []
  for entry in tqdm(data):
      prog_name = entry['prog_name']      
      func_name = entry['func_name']
      if not args.case_study:
        if not func_name.startswith('sub_'):
          continue
      # answer_and_probs = entry['answer_and_probs']
      logits = entry['logits']
      ids_answer = {}
      for k, v in logits.items():
        if k not in ids_answer:
          ids_answer[k] = []
        for name, score, prob in v:
          ids_answer[k].append((name, prob))        
        total_cnt += 1
      # for k,v in answer_and_probs.items():
      #   if k not in ids_answer:
      #     ids_answer[k] = []
      #   ids_answer[k].append((v, 1))
      #   total_cnt += 1

      entry['vars_answer'] = ids_answer
      for varname, gt_varname in list(entry['func_id_maps'].items()) + list(entry['var_id_maps'].items()):
        gt_varname = utils.try_demangle(gt_varname)
        if not utils.is_interesting_name(gt_varname):
          continue
        if gt_varname == varname:
          continue
        if varname not in entry['vars_answer']:
          pred_name = "<emptyname>"
        elif len(entry['vars_answer'][varname]) == 0:
          pred_name = "<emptyname>"
        elif entry['vars_answer'][varname][0][0] == '<unk>':
          pred_name = "<emptyname>"
        else:          
          for pred_name, _ in entry['vars_answer'][varname][:args.topk]:
            if pred_name == '<unk>':
              continue
            score = eval_utils.score_name(gt_varname, pred_name)
            ret.append(
              {
                'prog_name': prog_name,
                'func_name': func_name,
                # 'func_id': func_id,
                'varname': varname,
                'gt_varname': gt_varname,
                'pred_name': pred_name,
                'precision': score[0],
                'recall': score[1],
              }
            )
            
          
        if pred_name == "<emptyname>":
          score = (0, 0)
          ret.append({
            'prog_name': prog_name,
            'func_name': func_name,
            # 'func_id': func_id,
            'varname': varname,
            'gt_varname': gt_varname,
            'pred_name': pred_name,
            'precision': score[0],
            'recall': score[1],
          })

  return ret, error_cnt, total_cnt, dbg_error_entries, fixed_cnt


data_raw = open(args.fin, 'r').readlines()
data = []
for entry_str in data_raw:
  try:
    entry = json.loads(entry_str)
    data.append(entry)
  except:
    print("Error parsing json:", entry_str)
    continue

NUM_THREADS = 8
data_segs = []
for i in tqdm(range(NUM_THREADS)):
  data_segs.append(data[i::NUM_THREADS])


# ret = evaluate_batch(data_segs[0])

# multi-threaded
from multiprocessing import Pool

pool = Pool(NUM_THREADS)
rets = list(tqdm(pool.imap_unordered(evaluate_batch, data_segs), total=NUM_THREADS))
# rets = []
# for i in range(NUM_THREADS):
#   rets.append(evaluate_batch(data_segs[i]))

rets_all = []
error_cnt = 0
total_cnt = 0
total_fixed_cnt = 0
dbg_error_entries_all = []
for r, ec, tc, dbg_err_entries, fixed_cnt in rets:
  rets_all.extend(r)
  error_cnt += ec
  total_cnt += tc
  total_fixed_cnt += fixed_cnt
  dbg_error_entries_all.extend(dbg_err_entries)

print("Error cnt:", error_cnt)
print("Ratio of error %f=%d/%d" % (error_cnt / total_cnt, error_cnt, total_cnt))
print("Fixed cnt:", total_fixed_cnt)
if error_cnt > 0:
  print("Ratio of fixed %f=%d/%d" % (total_fixed_cnt / error_cnt, total_fixed_cnt, error_cnt))

print("Writing to file...")
fout = open(args.fout, 'w')
for line in tqdm(rets_all):
  fout.write(json.dumps(line) + '\n')
fout.close()

print("Writing to dbg file...")
fout = open(args.fout + '.dbg', 'w')
for entry, answer in tqdm(dbg_error_entries_all):
  fout.write(json.dumps({
    'entry': entry,
    'answer': answer,
  }) + '\n')
fout.close()
# dbg_funcs = {}
# for entry in data:
#     prog_name = entry['prog_name']
#     # func_id = md5(entry['func_body'].encode('utf-8')).hexdigest()
#     func_name = entry['func_name']
#     if (prog_name, func_name) not in dbg_funcs:
#         dbg_funcs[(prog_name, func_name)] = entry

# fout = open('dbg-funcs.jsonl', 'w')
# for (prog_name, func_name), func_entry in dbg_funcs.items():
#     fout.write(json.dumps({
#         'prog_name': prog_name,
#         'func_name': func_name,
#         'func_entry': func_entry,
#     }) + '\n')


print()