import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', type=str, required=True)
    parser.add_argument('--fout', type=str, default='')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.fout == '':
        args.fout = args.fin + '.preproc'

    with open(args.fin, 'r') as f:
        lines = f.readlines()

    # for each line, remove all "@<.*>"
    to_remove = re.compile(r'@<[0-9a-zA-Z]*>')
    preprocessed = []
    for line in lines:
        preprocessed.append(to_remove.sub('', line).replace('::', ''))
        
    with open(args.fout, 'w') as f:
        for line in preprocessed:
            f.write(line)

if __name__ == '__main__':
    main()
        