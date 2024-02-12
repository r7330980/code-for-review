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
    parser.add_argument("--config", type=str, default="openai-config.yaml")
    parser.add_argument("--progs", type=str, default="dirty-tiny-test-mysplit.preproc.pkl.0")
    parser.add_argument("--hints", type=str, default='')
    parser.add_argument("--from_idx", type=int, default=0)
    parser.add_argument("--to_idx", type=int, default=999999)
    parser.add_argument("--fout", type=str, default='tmp-zeroshot.jsonl')
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
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
        self.api_key = config.api_key
        self.func_to_ask = func
        self.var_id_maps = func.var_id_maps
        self.func_id_maps = func.func_id_maps        
        self.hint_string = hint_string        

    def get_ask_text(self):
        stripped_func_code = self.func_to_ask.body
        stripped_func_code = stripped_func_code[:self.config.max_func_strlen]
        ids = sorted(list(self.var_id_maps.keys())) + sorted(list(self.func_id_maps.keys()))
        ask_text = "\n\n"
        if self.hint_string != "":
            ask_text += self.hint_string + "\n"
        ask_text += "\nQ:["
        for var_id in ids[:self.config.max_ids]:
            ask_text += var_id + ","
        ask_text += "]\n"
        all_text = stripped_func_code + ask_text
        return all_text


    def cal_max_token_num(self):
        ids = sorted(list(self.var_id_maps.keys())) + sorted(list(self.func_id_maps.keys()))
        id_len = min(len(ids), self.config.max_ids)
        id_len_token = id_len * 20
        max_token_adjust = max(id_len_token, self.config.max_new_tokens)
        return max_token_adjust

    
    def ask_zeroshot(self):
        sys_prompt = """
You are a helpful binary program expert. You are helping the user to understand the binary program below.
You will suggest meaningful names for the variables and functions the user asks about.
The asked identifiers are specified in the format of Q:[var1,var2,...]
You will suggest one name for each asked identifier. 
You must output the suggested names in the json format:
{"var1": "suggested_name1", "var2": "suggested_name2", ...}
        """
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": self.get_ask_text()}
        ]
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=message,
            temperature=self.config.temperature,
            max_tokens=self.cal_max_token_num(),
            logprobs=False,
            stream=False,            
            n=self.config.num_ret_seq,
        )

        def parse_json_answer(answer):
            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            answer = answer.strip()
            answer = json.loads(answer)
            return answer
        answer_and_probs = []
        probs_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        cnt = 0
        ret_strs = []
        for choice in response.choices:
            answer = choice.message.content
            ret_strs.append(answer)
            try:
                answer = parse_json_answer(answer)
            except:
                answer = {'Error': answer}
            prob = probs_list[cnt]
            cnt += 1
            answer_and_probs.append((answer, prob))

        return {
            'func_name': self.func_to_ask.func_name,
            'func_body': self.func_to_ask.body,
            'ask_str': message,
            'answer_and_probs': answer_and_probs,
            'ret_strs': ret_strs,
            'var_id_maps': self.func_to_ask.var_id_maps,
            'func_id_maps': self.func_to_ask.func_id_maps,            
        }





def main():
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    api_key = config["api-key"]
    progs = pickle.load(open(args.progs, "rb"))
    if args.hints != "":
        hints_raw = [json.loads(l) for l in open(args.hints, 'r')]
        hints = {}
        for hint in hints_raw:
            prog_name = hint['prog_name']
            func_name = hint['func_name']
            hints[(prog_name, func_name)] = hint['hints']
        print("Loaded %d hints" % len(hints))
    else:
        hints = {}
    ask_config = AskConfig(
        api_key=api_key,
        max_new_tokens=100,
        max_ids=32,
        temperature=0.8,
        num_ret_seq=3,
        model=args.model,        
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
            if (prog.prog_name, func_name) in hints:
                hint_string = hints[(prog.prog_name, func_name)]
            else:
                hint_string = ""
            ask_manager = AskManager(ask_config, func_entry, hint_string)
            ret = ask_manager.ask_zeroshot()
            ret['prog_name'] = prog.prog_name
            fout.write(json.dumps(ret) + '\n')
            fout.flush()
    print()
    


if __name__ == "__main__":
    main()
