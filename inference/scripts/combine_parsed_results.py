# isort: off
import sys

sys.path.append("model")
sys.path.append("gennm_py_pb_parser")
# isort: on
import argparse
import pickle
import os
from tqdm import tqdm
import gennm_ir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fin",
        type=str,
        default="",
    )
    parser.add_argument("--in-dir", type=str, default="")
    parser.add_argument(
        "--fout",
        type=str,
        default="",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fout == "":
        args.fout = args.fin + ".combined"
    with open(args.fin, "rb") as f:
        data = pickle.load(f)
    for prog in tqdm(data):
        prog_name = prog.prog_name.split(".")[0]
        in_path = os.path.join(args.in_dir, prog_name)
        stripped_name2parsed = {}
        for func_id, func in prog.stripped_name2func.items():
            in_file = os.path.join(in_path, func_id + ".decomp.gennm.pb.pkl")
            if not os.path.exists(in_file):
                continue
            parsed = pickle.load(open(in_file, "rb"))
            stripped_name2parsed[func_id] = parsed
        prog.stripped_name2parsed = stripped_name2parsed

    with open(args.fout, "wb") as f:
        pickle.dump(data, f)
    print("Wrote to %s" % args.fout)

if __name__ == "__main__":
    main()
