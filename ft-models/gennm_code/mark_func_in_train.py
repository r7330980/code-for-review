import json
import gzip
import argparse
import pickle
import glob
import name_utils
from hashlib import sha256
from tqdm import tqdm
import datasets
import dump_func_in_train

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, default='dirty-test-mysplit')
parser.add_argument('--fout', type=str, default='dirty-test-in-train-new.json')
args = parser.parse_args()


test_files = glob.glob(args.test_dir + '/*')
print("Total test files: %d" % len(test_files))

in_train_hash_ds = datasets.load_from_disk('tmp-dirty-local-sha256-indices-map')

in_train_hash = set()
for entry in tqdm(in_train_hash_ds):
  in_train_hash.add(entry['sha256'])
print()


def mark_one_program(prog):
  prog_id = prog.prog_name.split('_')[0]
  name_func_map = prog.stripped_name2func
  in_train = set()
  for name, func in name_func_map.items():
    body = func.body.strip()
    renamed_body = body
    for var_id, gt_name in func.func_id_maps.items():
      if var_id.startswith('sub_'):
        demangled_gt_name = name_utils.try_demangle(gt_name)
        renamed_body = name_utils.replace_variable_names(
          renamed_body, var_id, demangled_gt_name)
    renamed_body = dump_func_in_train.addr_pattern.sub('\g<1>ADDR\g<3>', renamed_body)
    id = sha256(renamed_body.encode('utf-8')).hexdigest()
    if id in in_train_hash:
      in_train.add((prog_id, name, id, True, renamed_body))    
    else:
      in_train.add((prog_id, name, id, False, renamed_body))
  return in_train

def get_binaries(test_files):
  for test_file in test_files:
    test_data = pickle.load(open(test_file, 'rb'))
    for prog in test_data:
      yield prog

in_train_all = []
import multiprocessing as mp
pool = mp.Pool(24)
ret = tqdm(pool.imap_unordered(mark_one_program, get_binaries(test_files)))
for in_train in ret:
  in_train_all.extend(in_train)
pool.close()
pool.join()

import numpy as np

in_train_arr = np.array([itm[3] for itm in in_train_all])
print("Total in train: %d" % np.sum(in_train_arr))
print("In train ratio: %.4f" % (np.sum(in_train_arr) / len(in_train_arr)))

fout = open(args.fout, 'w')
fout.write(json.dumps(list(in_train_all)))
fout.close()




print()