# isort: off
import sys
sys.path.append("model")
# isort: on
from transformers import (
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    LlamaForCausalLM,
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    GenerationConfig
)
import gzip
import os
import transformers
import torch
import json
from tqdm import tqdm
import numpy as np
import name_utils
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx
import pickle
import copy

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
transformers.utils.logging.set_verbosity_info()
log_level = logging.INFO
logger.setLevel(log_level)

transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()



@dataclass
class GenNmArguments:
    model_id: Optional[str] = field(
        default="",
    )

    progs: Optional[str] = field(
        default="test_prog.preproc.pkl",
    )

    fout: Optional[str] = field(
        default="out_test_ask.jsonl",
    )

    from_idx: Optional[int] = field(
        default=0,
    )

    to_idx: Optional[int] = field(
        default=999999,
    )

    prop_callee_name: Optional[bool] = field(
        default=False,
    )
    dtype: Optional[str] = field(
        default="float16",
    )
    hint: Optional[str] = field(
        default="",
    )
    

class AskFTConfig(GenerationConfig):
    def __init__(self, 
                 prompt='prompt-ft.txt',
                 suffix=None,                 
                 max_func_strlen=5120,
                 max_ids=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.prompt = open(prompt, 'r').read()
        self.max_func_strlen = max_func_strlen
        self.max_ids = max_ids
        # self.prefix = "[INST]\n" + self.prompt + "\n"
        self.prefix = self.prompt + "\n"
        if suffix is None:
            self.suffix = """
            What would be a meaningful name for `%s`?
            [/INST]

            A meaningful name for `%s` would be `
            """.rstrip()
            self.answer_pattern = re.compile(r'A meaningful name for `[^`]*` would be `([^`]*)`')
        else:            
            self.suffix = suffix            



class AskFTManager:

    def __init__(self, config, func, hint_string):
        self.config = config
        self.func_to_ask = func
        self.var_id_maps = func.var_id_maps
        self.func_id_maps = func.func_id_maps   
        self.hint_string = hint_string             
    

    def ask_ids(self, model, tokenizer):
        stripped_func_code = self.func_to_ask.body
        stripped_func_code = stripped_func_code[:self.config.max_func_strlen]
        ids = sorted(list(self.var_id_maps.keys())) + sorted(list(self.func_id_maps.keys()))
        ask_text = "\n\n"
        if self.hint_string != "":
            ask_text += self.hint_string + "\n"
        ask_text += "\n\nQ:["
        for var_id in ids[:self.config.max_ids]:
            ask_text += var_id + ","
        ask_text = ask_text+ "]\n"
        ask_text += "Assistant: "
        all_text = self.config.prefix + stripped_func_code + ask_text
        ask_tokenized = tokenizer(all_text, return_tensors="pt")['input_ids'].cuda()
        max_new_token_adjusted = max(self.config.max_new_tokens,
                                    20+min(len(ids), self.config.max_ids)*20)
        # randomly sample an int from 0 to 1000
        rand_int = np.random.randint(0, 1000)

        set_seed(42)
        max_num_each_time = max(3072 // len(ask_tokenized[0]), 1)
        round_total = self.config.num_return_sequences // max_num_each_time + 1
        num_each_round = min(max_num_each_time, self.config.num_return_sequences)
        round_total = min(round_total, 3)
        ret_answers_all = []
        for rnd in range(round_total):            
            ret = model.generate(
                input_ids=ask_tokenized,
                max_new_tokens=max_new_token_adjusted,
                do_sample=self.config.do_sample,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                num_return_sequences=num_each_round,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=self.config.output_scores,
                return_dict_in_generate=True,
                # early_stopping=True,
            )
            score_ret = model.compute_transition_scores(ret.sequences, ret.scores, normalize_logits=True)
            # replace -inf with 0
            score_ret[score_ret == float('-inf')] = 0
            prob_ret = torch.exp(score_ret)
            # multiply along -1 axis
            prob_ret_red = torch.prod(prob_ret, dim=-1)
            input_id_len = ask_tokenized.shape[1]
            ret_decoded = tokenizer.batch_decode(ret.sequences[:, input_id_len:])   
            # detokenize
            detok_str = tokenizer.decode(ask_tokenized[0])                
            if rand_int > 900:
                fname = "dbg-%d.txt" % rand_int
                print("Writing dbg to %s" % fname)
                with open(fname, 'a') as f:
                    f.write("\n===========================\n")
                    f.write(detok_str)
                    f.write("\n\n")
                    f.write("\n\n".join(ret_decoded))
            ret_answers = [self.parse_answer(answer) for answer in ret_decoded]
            answer_score = [(a, s.item()) for a, s in zip(ret_answers, prob_ret_red) if a is not None]
            answer_score = sorted(answer_score, key=lambda x: x[1], reverse=True)
            ret_answers_all.extend(answer_score)
        return ret_answers_all, detok_str
        

    def parse_answer(self, answer):
        answer = answer.strip().rstrip('</s>').strip()
        try:
            ret = eval(answer)
            if not isinstance(ret, dict):
                ret = {'Error': answer}
            new_ret = {}
            for k, v in ret.items():
                if not ( isinstance(k, str) and isinstance(v, str)):
                    new_ret[str(k)] = str(v)
                else:
                    new_ret[k] = v
            ret = new_ret
        except:
            ret = {'Error': answer}
        return ret
        

def _prop_callee_name(func, known_names):
    ret_func = copy.deepcopy(func)
    new_func_body = ret_func.body
    to_remove = []
    for func_id in func.func_id_maps:
        if func_id == func.func_name:
            continue
        if func_id in known_names:
            new_name = known_names[func_id]
            new_func_body = name_utils.replace_variable_names(
                new_func_body, func_id, new_name)            
            ret_func.rename_map[new_name] = func_id
            to_remove.append(func_id)
    for func_id in to_remove:
        del ret_func.func_id_maps[func_id]            
    ret_func.body = new_func_body
    return ret_func


def ask_by_call_graph_topo(gennm_args, model, tokenizer, prog, fout, pbar, hints):
    call_graph = prog.call_graph
    stripped_name2func = prog.stripped_name2func
    # calculate out degree
    nodes_out_degree = []
    for node in call_graph.nodes:
        nodes_out_degree.append((node, call_graph.out_degree(node)))
    nodes_out_degree = sorted(nodes_out_degree, key=lambda x: x[1])
    known_names = {}
    for node in nodes_out_degree:
        if node[0] not in stripped_name2func:
            continue
        else:
            if node[1] > 0 and gennm_args.prop_callee_name:
                func_to_ask = _prop_callee_name(stripped_name2func[node[0]], known_names)
            else:
                func_to_ask = stripped_name2func[node[0]]
            if pbar.n < gennm_args.from_idx:
                pbar.update(1)
                continue
            if pbar.n > gennm_args.to_idx:
                break
            pbar.update(1)
            ask_config = AskFTConfig(                
                max_new_tokens=100,
                max_ids=32,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=3,
                output_scores=True,
                return_dict_in_generate=True,
                suffix=""
            )
            prog_name = prog.prog_name
            func_name = func_to_ask.func_name
            if (prog_name, func_name) in hints and gennm_args.hint != "":
                hint_string = hints[prog_name, func_name]
            else:                
                hint_string = ""
                if gennm_args.hint != "":
                    print("Hint not found for %s %s" % (prog_name, func_name))
            ask_manager = AskFTManager(ask_config, func_to_ask, hint_string=hint_string)
            answer_and_probs, ask_str = ask_manager.ask_ids(model, tokenizer)
            if func_to_ask.func_name in answer_and_probs[0][0] and func_to_ask.func_name not in known_names:
                known_names[func_to_ask.func_name] = answer_and_probs[0][0][func_to_ask.func_name]
            ret = {
                'prog_name': prog.prog_name,
                'func_name': func_to_ask.func_name,
                'func_body': func_to_ask.body,
                'rename_map': func_to_ask.rename_map,
                'var_id_maps': func_to_ask.var_id_maps,
                'func_id_maps': func_to_ask.func_id_maps,
                'answer_and_probs': answer_and_probs,
                'ask_str': ask_str
            }
            fout.write(json.dumps(ret) + '\n')
            fout.flush()

           


def main():
    parser = HfArgumentParser((GenNmArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        gennm_args = parser.parse_yaml_file(
            json_file=os.path.abspath(sys.argv[1])[0]
        )
    else:
        gennm_args = parser.parse_args_into_dataclasses()[0]
    logger.info(gennm_args)

    with open(gennm_args.progs, "rb") as f:
        progs = pickle.load(f)
    logger.info("Loaded %d progs" % len(progs))
    model_id = gennm_args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16
    if gennm_args.dtype == "float32":
        dtype = torch.float32
    elif gennm_args.dtype == "bfloat16":
        dtype = torch.bfloat16    
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        # torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
        torch_dtype=dtype,
    ).to("cuda")
    fout = open(gennm_args.fout, 'w')
    total_funcs = 0
    for prog in progs:
        total_funcs += len(prog.stripped_name2func)
    pbar = tqdm(total=min(total_funcs, gennm_args.to_idx))
    logger.setLevel(logging.WARN)
    transformers.utils.logging.set_verbosity(logging.WARN)
    if gennm_args.hint != "":
        hints = {}
        hints_raw = [json.loads(l) for l in open(gennm_args.hint, 'r')]
        for entry in hints_raw:
            prog_name = entry['prog_name']
            func_name = entry['func_name']
            hints[prog_name, func_name] = entry['hints']
        print("Loaded %d hints" % len(hints))
    else:
        hints = {}

    with torch.no_grad():        
        for prog in progs:                        
            if pbar.n > gennm_args.to_idx:
                break
            ask_by_call_graph_topo(gennm_args, model, tokenizer, prog, fout, pbar, hints)    
    

if __name__ == "__main__":
    main()

