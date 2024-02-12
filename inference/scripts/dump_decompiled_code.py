# isort: off
import sys

sys.path.append("model")
# isort: on
import argparse
import pickle
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fin",
        type=str,
        default="dirty-tiny-test-mysplit/dirty-tiny-test-mysplit.preproc.pkl.0",
    )
    parser.add_argument("--out-dir", type=str, default="dirty-tiny-test-mysplit-decomp")
    parser.add_argument("--def-in", type=str, default="defs.h")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.fin, "rb") as f:
        data = pickle.load(f)
    for prog in tqdm(data):
        prog_name = prog.prog_name.split(".")[0]
        out_path = os.path.join(args.out_dir, prog_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            # cp def_in to out_path
            os.system("cp {} {}".format(args.def_in, out_path))

        for func_id, func in prog.stripped_name2func.items():
            func_body = func.body
            out_text = '#include "defs.h"\n\n' + func_body
            out_file = os.path.join(out_path, func_id + ".decomp.c")
            with open(out_file, "w") as f:
                f.write(out_text)

    print()


if __name__ == "__main__":
    main()
