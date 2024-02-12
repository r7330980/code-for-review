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
parser.add_argument('--in-train', type=str, default='')
parser.add_argument('--intrain-name', type=str, default='')
parser.add_argument('--dirty-data', type=str, default='')
parser.add_argument('--exact-match', action='store_true')
args = parser.parse_args()

func_name_pattern = re.compile(r"[^0-9a-zA-Z_]sub_[0-9a-fA-F_]+[^0-9a-zA-Z_]")
if args.dirty_data == "":
    dirty_entry_dict = {}
else:
    print("Loading dirty data...")
    dirty_entries = open(args.dirty_data, 'r').readlines()
    dirty_entries = [json.loads(line) for line in dirty_entries]
    dirty_entry_dict = {}
    for entry in dirty_entries:
        bin_name = entry['binary']
        prog_name = "%s_%s.jsonl.gz" %(bin_name, bin_name)
        func_name = entry['funcname']
        var_name = entry['ori_name']
        if (prog_name, func_name, var_name) not in dirty_entry_dict:
            dirty_entry_dict[(prog_name, func_name, var_name)] = entry


in_train_name = set()
in_train_name_data = json.load(open(args.intrain_name, 'r'))
for entry in tqdm(in_train_name_data):
    in_train_name.add(entry)

in_train_set = set()
in_train_data = json.load(open(args.in_train, 'r'))
not_in_train_set = set()
for entry in tqdm(in_train_data):
    prog_name = entry[0]
    func_name = entry[1]
    is_in_train = entry[3]
    if is_in_train:
        in_train_set.add((prog_name, func_name))
    else:
        not_in_train_set.add((prog_name, func_name))
# leaf_funcs = set()
# for func_id, entry in dbg_func_id2entry.items():
#     body = entry["func_body"]
#     # count number of func_name_pattern
#     num_func_name = len(func_name_pattern.findall(body))
#     if num_func_name <= 1:
#         leaf_funcs.add(func_id)


# ret_fin = open('tmp-eval-first-name-out.jsonl', 'r')
# ret_fin = open('tmp-eval-highest-name-out.jsonl', 'r')
# ret_fin = open('tmp-eval-prop-callee-first.jsonl', 'r')
# ret_fin = open("tmp-eval-compare-first.jsonl", "r")
# ret_fin = open('tmp-eval-prop-callee-highest.jsonl', 'r')
# ret_fin = open('tmp-eval-compare-highest.jsonl', 'r')
# ret_fin = open('tmp-eval-ft-best.jsonl', 'r')
ret_fin = open(args.ret_in, 'r')
# ret_fin = open('tmp-eval-ft-7200-best.jsonl', 'r')
# ret_fin = open('tmp-eval-ft-7200.jsonl', 'r')
data_raw = [json.loads(line) for line in ret_fin.readlines()]
ret_fin.close()
data = []
seen = set()
dup = 0
for entry in tqdm(data_raw):
    prog_name = entry['prog_name']
    func_name = entry['func_name']
    ori_varname = entry['varname']
    if (prog_name, func_name, ori_varname) in seen:
        dup += 1
        continue
    seen.add((prog_name, func_name, ori_varname))
    data.append(entry)

print("Number of duplicates: %d (%.2f)" % (dup, dup / len(data_raw)))


problematic = []
for entry in tqdm(data):
    prog_name = entry['prog_name'].split('_')[0]
    func_name = entry['func_name']
    if (prog_name, func_name) in in_train_set:
        continue
    if (prog_name, func_name) in not_in_train_set:
        continue
    problematic.append(entry)


num_programs = len(set([entry["prog_name"] for entry in data]))
num_funcs = len(set([(entry["prog_name"], entry["func_name"]) for entry in data]))
num_names = len(data)
print(
    "Dataset stats: %d programs, %d functions, %d names"
    % (num_programs, num_funcs, num_names)
)

# average precision and recall
if not args.exact_match:
    precisions = np.array([entry["precision"] for entry in data])
    recalls = np.array([entry["recall"] for entry in data])
else:
    print("Exact match")
    precisions = []
    recalls = []
    for entry in data:
        if entry['pred_name'] == entry['gt_varname']:
            precisions.append(1)
            recalls.append(1)
        else:
            precisions.append(0)
            recalls.append(0)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    print("Average precision: %.4f" % (precisions.mean()))
    print("Average recall: %.4f" % (recalls.mean()))

is_var_name = np.array(
    [not entry["varname"].startswith("sub_") for entry in data]
)

is_in_dirty = []
if len(dirty_entry_dict) == 0:
    is_in_dirty = np.array([True for entry in data])
else:
    for entry in data:
        prog_name = entry['prog_name']
        func_name = entry['func_name']
        var_name = entry['varname']
        if (prog_name, func_name, var_name) in dirty_entry_dict:
            is_in_dirty.append(True)
        else:
            is_in_dirty.append(False)

is_in_dirty = np.array(is_in_dirty)
is_var_name &= is_in_dirty

is_non_empty_name = np.array(
    [entry["pred_name"] != "<emptyname>" for entry in data]
)
is_func_name = np.array(
    [entry["varname"].startswith("sub_") and entry["varname"] == entry["func_name"] for entry in data]
)

is_in_train = []
is_name_in_train = []
for entry in data:
    prog_name = entry["prog_name"].split('_')[0]
    func_name = entry["func_name"]
    if (prog_name, func_name) in in_train_set:
        is_in_train.append(True)
    else:
        is_in_train.append(False)
    if entry['gt_varname'] in in_train_name:
        is_name_in_train.append(True)
    else:
        is_name_in_train.append(False)
is_in_train = np.array(is_in_train)
is_name_in_train = np.array(is_name_in_train)

print("Number of in-train: %d (%.2f)" % (is_in_train.sum(), is_in_train.sum() / len(is_in_train)))
print("Number of not in-train: %d (%.2f)" % ((~is_in_train).sum(), (~is_in_train).sum() / len(is_in_train)))

var_precisions = precisions[is_var_name]
var_recalls = recalls[is_var_name]
var_precisions_fit = precisions[is_var_name & is_in_train]
var_recalls_fit = recalls[is_var_name & is_in_train]
var_precisions_fnit = precisions[is_var_name & (~is_in_train)]
var_recalls_fnit = recalls[is_var_name & (~is_in_train)]

var_precisions_name_it = precisions[is_var_name & is_name_in_train]
var_recalls_name_it = recalls[is_var_name & is_name_in_train]
var_precisions_name_nit = precisions[is_var_name & (~is_name_in_train)]
var_recalls_name_nit = recalls[is_var_name & (~is_name_in_train)]

non_emptry_var_precisions_fit = precisions[is_var_name & is_non_empty_name & is_in_train]
non_emptry_var_recalls_fit = recalls[is_var_name & is_non_empty_name & is_in_train]

non_emptry_var_precisions_fnit = precisions[is_var_name & is_non_empty_name & (~is_in_train)]
non_emptry_var_recalls_fnit = recalls[is_var_name & is_non_empty_name & (~is_in_train)]

overall_non_empty_precisions = precisions[is_non_empty_name]
overall_non_empty_recalls = recalls[is_non_empty_name]



# non_emptry_precisions = precisions[is_non_empty_name]
# non_emptry_recalls = recalls[is_non_empty_name]

func_precisions = precisions[is_func_name]
func_recalls = recalls[is_func_name]

func_precisions_fit = precisions[is_func_name & is_in_train]
func_recalls_fit = recalls[is_func_name & is_in_train]
func_precisions_fnit = precisions[is_func_name & (~is_in_train)]
func_recalls_fnit = recalls[is_func_name & (~is_in_train)]

non_emptry_func_precisions = precisions[is_func_name & is_non_empty_name]
non_emptry_func_recalls = recalls[is_func_name & is_non_empty_name]

print("======Variables=========")
print("#: %d (%.2f)" % (len(var_precisions), len(var_precisions) / len(precisions)))
pr_overall = var_precisions.mean()
rc_overall = var_recalls.mean()
print("Precision: %.4f" % (pr_overall))
print("Recall: %.4f" % (rc_overall))
print('---------------------------------')
pr_fit = var_precisions_fit.mean()
rc_fit = var_recalls_fit.mean()
pr_fnit = var_precisions_fnit.mean()
rc_fnit = var_recalls_fnit.mean()
print("Precision fit: %.4f" % (pr_fit))
print("Recall fit: %.4f" % (rc_fit))
print("Precision fnit: %.4f" % (pr_fnit))
print("Recall fnit: %.4f" % (rc_fnit))
print('---------------------------------')
print("Ground truth in-train: %d (%.2f)" % ((is_name_in_train & is_var_name).sum(), (is_name_in_train & is_var_name).sum() / is_var_name.sum()))
print("Ground truth not-in-train: %d (%.2f)" % ((~is_name_in_train & is_var_name).sum(), (~is_name_in_train & is_var_name).sum() / is_var_name.sum()))
print("Precision gt not-in-train: %.4f" % (var_precisions_name_nit.mean()))
print("Recall gt not-in-train: %.4f" % (var_recalls_name_nit.mean()))
print("Precision gt in-train: %.4f" % (var_precisions_name_it.mean()))
print("Recall gt in-train: %.4f" % (var_recalls_name_it.mean()))
print('---------------------------------')
all_non_empty_len = len(non_emptry_var_precisions_fit) + len(non_emptry_var_precisions_fnit)
print("#Non-empty: %d (%.2f)" % (all_non_empty_len, all_non_empty_len / len(var_precisions)))
pr_fit_ne = non_emptry_var_precisions_fit.mean()
rc_fit_ne = non_emptry_var_recalls_fit.mean()
pr_fnit_ne = non_emptry_var_precisions_fnit.mean()
rc_fnit_ne = non_emptry_var_recalls_fnit.mean()
print("Non-empty precision fit: %.4f" % (pr_fit_ne))
print("Non-empty recall fit: %.4f" % (rc_fit_ne))
print("Non-empty precision fnit: %.4f" % (pr_fnit_ne))
print("Non-empty recall fnit: %.4f" % (rc_fnit_ne))
pr_overall_ne = overall_non_empty_precisions.mean()
rc_overall_ne = overall_non_empty_recalls.mean()
print("Non-empty precision overall: %.4f" % (pr_overall_ne))
print("Non-empty recall overall: %.4f" % (rc_overall_ne))
print("#In-train: %d (%.2f)" % ((is_in_train & is_var_name).sum(), (is_in_train & is_var_name).sum() /is_var_name.sum()))
print("#Not in-train: %d (%.2f)" % ((is_var_name & ~is_in_train).sum(), (is_var_name & ~is_in_train).sum() / is_var_name.sum()))
print("PR-FIT,RC-FIT,PR-FNIT,RC-FNIT,PR-OVERALL,RC-OVERALL,PR-FIT-NE,RC-FIT-NE,PR-FNIT-NE,RC-FNIT-NE,PR-OVERALL-NE,RC-OVERALL-NE")
print("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (pr_fit, rc_fit, pr_fnit, rc_fnit, pr_overall, rc_overall, pr_fit_ne, rc_fit_ne, pr_fnit_ne, rc_fnit_ne, pr_overall_ne, rc_overall_ne))


our_set = set()
for entry in data:
    prog_name = entry['prog_name']
    func_name = entry['func_name']
    var_name = entry['varname']
    our_set.add((prog_name, func_name, var_name))

# count only in dirty as wrong
only_in_dirty_in_train_cnt = 0
only_in_dirty_not_in_train_cnt = 0
only_in_dirty = []
for k in dirty_entry_dict.keys():
    if k not in our_set:
        only_in_dirty.append(k)
        if k in in_train_set:
            only_in_dirty_in_train_cnt += 1
        else:
            only_in_dirty_not_in_train_cnt += 1

print("Number of only in dirty: %d (%.2f)" % (len(only_in_dirty), len(only_in_dirty) / len(our_set)))

# count only in dirty as wrong
new_total = len(var_precisions_fit) + only_in_dirty_in_train_cnt
new_pr_fit = pr_fit * len(var_precisions_fit) / new_total + 0 * only_in_dirty_in_train_cnt / new_total
new_rc_fit = rc_fit * len(var_precisions_fit) / new_total + 0 * only_in_dirty_in_train_cnt / new_total
new_total = len(var_precisions_fnit) + only_in_dirty_not_in_train_cnt
new_pr_fnit = pr_fnit * len(var_precisions_fnit) / new_total + 0 * only_in_dirty_not_in_train_cnt / new_total
new_rc_fnit = rc_fnit * len(var_precisions_fnit) / new_total + 0 * only_in_dirty_not_in_train_cnt / new_total
new_total = len(var_precisions) + len(only_in_dirty)
new_pr = pr_overall * len(var_precisions) / new_total + 0 * len(only_in_dirty) / new_total
new_rc = rc_overall * len(var_precisions) / new_total + 0 * len(only_in_dirty) / new_total
print("============Count only in dirty as wrong============")
print("New PR-FIT,RC-FIT,PR-FNIT,RC-FNIT,PR-OVERALL,RC-OVERALL")
print("%f,%f,%f,%f,%f,%f" % (new_pr_fit, new_rc_fit, new_pr_fnit, new_rc_fnit, new_pr, new_rc))


print("======Functions=========")
print("#: %d (%.2f)" % (len(func_precisions), len(func_precisions) / len(precisions)))
print("Precision: %.4f" % (func_precisions.mean()))
print("Recall: %.4f" % (func_recalls.mean()))
print('---------------------------------')
print("Precision fit: %.4f" % (func_precisions_fit.mean()))
print("Recall fit: %.4f" % (func_recalls_fit.mean()))
print("Precision fnit: %.4f" % (func_precisions_fnit.mean()))
print("Recall fnit: %.4f" % (func_recalls_fnit.mean()))
print('---------------------------------')
print("#Non-empty: %d (%.2f)" % (len(non_emptry_func_precisions), len(non_emptry_func_precisions) / len(func_precisions)))
print("Non-empty precision: %.4f" % (non_emptry_func_precisions.mean()))
print("Non-empty recall: %.4f" % (non_emptry_func_recalls.mean()))
