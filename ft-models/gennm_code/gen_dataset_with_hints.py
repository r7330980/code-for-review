import json
import argparse
import gzip
from tqdm import tqdm
import name_utils
import gzip
from binary_prog import Function, BinaryProgram
import numpy as np
import cxxfilt
import re
import datasets
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hints", type=str, default="hints.jsonl")
    parser.add_argument("--binary", type=str, default="hints_training_binaries.txt")
    parser.add_argument("--fout", type=str, default="data/dirty-mysplit-hints-train")
    parser.add_argument("--shard", action="store_true")
    args = parser.parse_args()
    return args


def generate_split(args, binaries, fout_name, shard, hints, prompt):
    if len(binaries) == 0:
        return
    shard_cnt = 0
    instance_cnt = 0
    if shard:
        fout = open(fout_name + ".%d" % shard_cnt, "w")
    else:
        fout = open(fout_name, "w")
    for current_prog in tqdm(binaries):        
        prog = current_prog
        funcs = prog.stripped_name2func
        for name, func in funcs.items():
            ds_entry = {}
            ds_entry["prog_name"] = prog.prog_name
            ds_entry["strip_func_name"] = name
            ds_entry[
                "prompt"
            ] = prompt
            query_string = "\n\n\n"
            query_string += func.body
            query_string += "\n\n\n"
            if (prog.prog_name, name) in hints:
                hints_string = str(hints[(prog.prog_name, name)])
                query_string += hints_string + "\n"
                # ds_entry["query"] = query_string + "\n" + hints_string
            else:
                print("No hints for (%s, %s)" % (prog.prog_name, name))
                hints_string = ""
                # ds_entry["query"] = query_string
            ret = {}
            query_string += "Q:["
            for var_id, gt_name in func.var_id_maps.items():
                query_string += "%s," % var_id
                ret[var_id] = gt_name
            for func_id, gt_name in func.func_id_maps.items():
                query_string += "%s," % func_id
                ret[func_id] = name_utils.try_demangle(gt_name)
            query_string += "]\nAssistant: "
            ds_entry["query"] = query_string
            response_str = str(ret)
            ds_entry["response"] = response_str
            fout.write(json.dumps(ds_entry) + "\n")
            instance_cnt += 1
            if shard and instance_cnt % 100000 == 0:
                shard_cnt += 1
                fout.close()
                fout = open(fout_name + ".%d" % shard_cnt, "w")
            fout.flush()

    fout.close()


def main():
    args = parse_args()
    print("Loading hints...")
    hints = [json.loads(l) for l in open(args.hints, "r").readlines()]
    prog_func2hints = {}
    for hint in hints:
        prog_name = hint["prog_name"]
        func_name = hint["func_name"]
        prog_func2hints[(prog_name, func_name)] = hint["hints"]

    print("Loading binaries...")
    binary_flist = [l.strip() for l in open(args.binary, "r").readlines()]
    binary_progs = []
    for binary_file in tqdm(binary_flist):
        binary_progs.extend(pickle.load(open(binary_file, "rb")))

    prompt = open('prompt-ft.txt').read()
    print("Prompt: %s" % prompt)
    np.random.seed(42)
    fout = open(args.fout + ".flist.train", "w")
    for binary in binary_progs:
        fout.write(binary.prog_name + "\n")
    fout.close()

    print("Generating training...")
    generate_split(args, binary_progs, args.fout + ".train", args.shard, hints=prog_func2hints, prompt=prompt)
    print("Generating validation...")


if __name__ == "__main__":
    main()
