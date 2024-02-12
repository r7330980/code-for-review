import json
import argparse
import pickle
import os
from tqdm import tqdm
import multiprocessing
import functools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ret", type=str, default="prop-rets/malware-prop-top3-names-rnd0.jsonl"
    )
    parser.add_argument("--binary", type=str, default="valid-flist.txt")
    parser.add_argument("--max-hints-num", type=int, default=20)
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument("--fout", type=str, default="hints.jsonl")
    args = parser.parse_args()
    return args


def process_one_binary(binary, prog_func2entry, prog_ori2entry, args):
    ret = {}
    funcs = binary.stripped_name2func
    call_graph = binary.call_graph
    for func_name, func_entry in funcs.items():
        callers = set()
        callees = set()
        for caller, callee in call_graph.edges:
            if callee == func_name:
                callers.add(caller)
            if caller == func_name:
                callees.add(callee)
        caller_hints = []
        callee_hints = []
        my_occurs = []
        for caller in callers:
            if (binary.prog_name, caller) in prog_func2entry:
                caller_hints.extend(prog_func2entry[(binary.prog_name, caller)])
        for callee in callees:
            if (binary.prog_name, callee) in prog_func2entry:
                callee_hints.extend(prog_func2entry[(binary.prog_name, callee)])
        if (binary.prog_name, func_name) in prog_ori2entry:
            my_occurs.extend(prog_ori2entry[(binary.prog_name, func_name)])
        if args.ground_truth:
            caller_hints = [h["gt_varname"] for h in caller_hints]
            callee_hints = [h["gt_varname"] for h in callee_hints]
            my_occurs = [h["gt_varname"] for h in my_occurs]
        else:
            caller_hints = [h["pred_name"] for h in caller_hints]
            callee_hints = [h["pred_name"] for h in callee_hints]
            my_occurs = [h["pred_name"] for h in my_occurs]
        ret[(binary.prog_name, func_name)] = {
            "caller_hints": caller_hints[: args.max_hints_num],
            "callee_hints": callee_hints[: args.max_hints_num],
            "my_occurs": my_occurs[: args.max_hints_num],
        }
    return ret


def process_binary_batch(binary_batch, prog_func2entry, prog_ori2entry, args):
    rets = []
    for binary in tqdm(binary_batch):
        rets.append(process_one_binary(binary, prog_func2entry, prog_ori2entry, args))
    return rets


def main():
    args = parse_args()
    print("Loading model rets...")
    model_rets = [json.loads(l.strip()) for l in open(args.model_ret, "r").readlines()]
    prog_func2entry = {}
    prog_ori2entry = {}
    for entry in model_rets:
        prog_name = entry["prog_name"]
        func_name = entry["func_name"]
        ori_var_name = entry["varname"]
        if (prog_name, func_name) not in prog_func2entry:
            prog_func2entry[(prog_name, func_name)] = []
        prog_func2entry[(prog_name, func_name)].append(entry)
        if (prog_name, ori_var_name) not in prog_ori2entry:
            prog_ori2entry[(prog_name, ori_var_name)] = []
        prog_ori2entry[(prog_name, ori_var_name)].append(entry)

    print("Loading binaries...")
    binary_flist = [l.strip() for l in open(args.binary, "r").readlines()]
    binary_progs = []
    for binary_file in tqdm(binary_flist):
        binary_progs.extend(pickle.load(open(binary_file, "rb")))

    NUM_WORKERS = 4
    binary_batches = []
    batch_size = len(binary_progs) // NUM_WORKERS
    for i in range(NUM_WORKERS):
        binary_batches.append(binary_progs[i * batch_size : (i + 1) * batch_size])
    binary_batches[-1].extend(binary_progs[NUM_WORKERS * batch_size :])

    print("Processing binaries...")
    pool = multiprocessing.Pool(NUM_WORKERS)
    process_binary_batch_partial = functools.partial(
        process_binary_batch,
        prog_func2entry=prog_func2entry,
        prog_ori2entry=prog_ori2entry,
        args=args,
    )
    ret_all = []
    for ret in pool.imap_unordered(
        process_binary_batch_partial, binary_batches, chunksize=1
    ):
        ret_all.extend(ret)

    # process_one_binary_partial = functools.partial(process_one_binary, prog_func2entry=prog_func2entry, prog_ori2entry=prog_ori2entry, args=args)
    # pool = multiprocessing.Pool(24)
    # ret_all = [i for i in tqdm(pool.imap_unordered(process_one_binary_partial, binary_progs, chunksize=1),total=len(binary_progs))]

    print("Writing to %s..." % args.fout)
    fout = open(args.fout, "w")
    for ret in tqdm(ret_all):
        for k, v in ret.items():
            hint_string = ""
            hint_string += "Hints from callers: ["
            hint_string += ", ".join(v["caller_hints"])
            hint_string += "]\n"
            hint_string += "Hints from callees: ["
            hint_string += ", ".join(v["callee_hints"])
            hint_string += "]\n"
            hint_string += "Hints from callsites: ["
            hint_string += ", ".join(v["my_occurs"])
            hint_string += "]\n"

            fout.write(
                json.dumps({"prog_name": k[0], "func_name": k[1], "hints": hint_string})
                + "\n"
            )

    fout.close()

    print()


if __name__ == "__main__":
    main()
