# isort: off
import sys

sys.path.append("model")
# isort: on
import argparse
import json
import yaml
from binary_prog import BinaryProgram, Function
import pickle
import openai
from openai import OpenAI
from transformers import (GenerationConfig)
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--progs", type=str, default="dirty-train-mysplit/dirty-train-mysplit.preproc.pkl.0")
    parser.add_argument("--hints", type=str, default='')
    parser.add_argument("--from_idx", type=int, default=0)
    parser.add_argument("--to_idx", type=int, default=999999)
    parser.add_argument("--fout", type=str, default='tmp-fewshot_examples.jsonl')    
    args = parser.parse_args()
    return args


class AskConfig(GenerationConfig):
    def __init__(self, api_key, max_func_strlen=5120, max_ids=10, 
                 model="gpt-3.5-turbo-1106",                 
                 **kwargs):
        super().__init__(**kwargs)
        self.max_func_strlen = max_func_strlen
        self.max_ids = max_ids
        self.model = model
        self.api_key = api_key

class AskManager:

    def __init__(self, config, func, hint_string):
        self.config = config        
        self.func_to_ask = func
        self.var_id_maps = func.var_id_maps
        self.func_id_maps = func.func_id_maps        
        self.hint_string = hint_string        

    def get_ask_text_and_answer(self):
        stripped_func_code = self.func_to_ask.body
        stripped_func_code = stripped_func_code[:self.config.max_func_strlen]
        all_id_maps = {}
        all_id_maps.update(self.var_id_maps)
        all_id_maps.update(self.func_id_maps)
        ids = sorted(list(self.var_id_maps.keys())) + sorted(list(self.func_id_maps.keys()))
        ask_text = "\n\n"
        if self.hint_string != "":
            ask_text += self.hint_string + "\n"
        ask_text += "\nQ:["
        for var_id in ids[:self.config.max_ids]:
            ask_text += var_id + ","
        ask_text += "]\n"
        all_text = stripped_func_code + ask_text
        answer_text = "```json\n{"
        for var_id in ids[:self.config.max_ids]:
            answer_text += '"%s":"%s",' % (var_id, all_id_maps[var_id])
        if len(ids) > 0:
            answer_text = answer_text[:-1]            
        answer_text += "}\n```"
        return all_text, answer_text






def main():
    args = parse_args()
    # config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    # api_key = config["api-key"]
    progs = pickle.load(open(args.progs, "rb"))
    # if args.hints != "":
    #     hints_raw = [json.loads(l) for l in open(args.hints, 'r')]
    #     hints = {}
    #     for hint in hints_raw:
    #         prog_name = hint['prog_name']
    #         func_name = hint['func_name']
    #         hints[(prog_name, func_name)] = hint['hints']
    #     print("Loaded %d hints" % len(hints))
    # else:
    #     hints = {}
    ask_config = AskConfig(
        api_key="",
        max_new_tokens=100,
        max_ids=32,
        temperature=0.8,
        num_ret_seq=3,
        model="",        
    )
    pbar = tqdm(total=args.to_idx)
    fout = open(args.fout, 'w')
    instance_cnt = 0
    for prog in progs:
        for func_name, func_entry in prog.stripped_name2func.items():
            pbar.update(1)
            if instance_cnt < args.from_idx:
                instance_cnt += 1
                continue
            if instance_cnt >= args.to_idx:
                break
            instance_cnt += 1            
            hint_string = ""
            ask_manager = AskManager(ask_config, func_entry, hint_string)
            ask, answer = ask_manager.get_ask_text_and_answer()
            ret = {
                'prog_name': prog.prog_name,
                'func_name': func_name,
                'ask': ask,
                'answer': answer,
            }
            fout.write(json.dumps(ret) + '\n')
            fout.flush()
    print()
    


if __name__ == "__main__":
    main()
