import os
import tiktoken
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union, Optional
import torch
from datasets import load_from_disk

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from llmtuner.extras.template import get_template_and_fix_tokenizer

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer
    from llmtuner.hparams import DataArguments, GenNmArguments


logger = get_logger(__name__)


def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "lc"],
    gennm_args: Optional["GenNmArguments"] = None,
    str2id: Optional[Dict[str, int]] = None,
    id2str: Optional[Dict[int, str]] = None,
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def construct_gennm_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            # query = query + "\nAssistant:"
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            # if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
            if not (isinstance(query, str) and isinstance(response, str) and (query != "" or response != "")):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(data_args.cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(data_args.cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs
    
    def preprocess_gennm_classification_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        model_inputs = {"input_ids": [], "attention_mask": [], "clm_labels": [], "cls_labels": []}
        if 'GPT2' in str(type(tokenizer)):
            interesting_id_first = 950
            interesting_ids = set(
                [7610, 330, 11477, 28948]
            )
        elif 'llama' in str(type(tokenizer)).lower():
            interesting_id_first = 2396
            interesting_ids = set(
                [525, 22868]
            )
        # tokenizer.encode("a': '") --> [1, 263, 2396, 525]
        # 2396, 525 is the pattern <':>  and  < '>
        # the token after 2396 525 is then the variable name we want to classify
        # or 2396 22868, where 22868 is < '_>
        for query, response, history, system in construct_gennm_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, clm_labels, cls_labels = [], [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query=query, 
                resp=response, 
                history=history, 
                system=system
            )):                              
                max_source_len = int(data_args.cutoff_len - len(target_ids))
                max_target_len = len(target_ids)
                if max_source_len < 0:
                    max_source_len = 1
                
                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                

                if data_args.train_on_prompt:
                    source_mask = source_ids
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)        
                input_ids += source_ids + target_ids
                clm_labels += source_mask + target_ids
                # Note: python dict does not have order
                # Cannot guarantee the behavior if we directly "eval" the string
                content = response.replace("{", "").replace("}", "")
                target_answer = []
                for old_name2new_name in content.split(','):
                    if len(old_name2new_name.split("': ")) != 2:
                        print(old_name2new_name)
                        print(content)
                        exit(0)
                    old_name, new_name = old_name2new_name.split("': ")
                    old_name = old_name.strip().strip("'")
                    if old_name.startswith('sub_'):
                        target_answer.append("GENNM_SKIP")
                    else:
                        new_name = new_name.strip().strip("'")                    
                        target_answer.append(new_name)
                # print("target_answer: ", target_answer)
                target_mask = [IGNORE_INDEX] * len(target_ids)
                STATE_NO = 0
                STATE_SEEN_2396 = 1
                STATE_INTERESTED = 2
                pattern_state = STATE_NO
                answer_cnt = 0
                for i, token_id in enumerate(target_ids):
                    if token_id == interesting_id_first:
                        pattern_state = STATE_SEEN_2396
                    elif pattern_state == STATE_SEEN_2396:
                        # if token_id == 525 or token_id == 22868:
                        if token_id in interesting_ids:
                            pattern_state = STATE_INTERESTED
                        else:
                            pattern_state = STATE_NO
                    elif pattern_state == STATE_INTERESTED:
                        next_answer = target_answer[answer_cnt]
                        answer_cnt += 1
                        if 'GENNM_SKIP' in next_answer:
                            pass
                            # target_mask[i] = IGNORE_INDEX
                        elif next_answer in str2id:
                            target_mask[i] = str2id[next_answer]
                        else:
                            target_mask[i] = str2id["<unk>"]
                        pattern_state = STATE_NO
                if answer_cnt != len(target_answer):
                    print("ERROR: answer_cnt != len(target_answer)")
                    print("ERROR: answer_cnt: ", answer_cnt)
                    print("ERROR: len(target_answer): ", len(target_answer))
                    print("ERROR: target_answer: ", target_answer)
                cls_labels += [IGNORE_INDEX] * len(source_ids) + target_mask
        
            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                clm_labels = clm_labels[:data_args.cutoff_len]
                cls_labels = cls_labels[:data_args.cutoff_len]
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["clm_labels"].append(clm_labels)
            model_inputs["cls_labels"].append(cls_labels)
        return model_inputs
                


    def preprocess_packed_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                tokenizer, query, response, history, system
            )):
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i: i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i: i + block_size])

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and query != ""):
                continue

            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):            
            if not (isinstance(query, str) and isinstance(response, list) and query != "" and len(response) > 1):
                continue

            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
            max_source_len = int(data_args.cutoff_len * (len(prompt_ids) / total_len))
            max_target_len = int(data_args.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

            if len(prompt_ids) > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if len(chosen_ids) > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
            if len(rejected_ids) > max_target_len:
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    def print_gennm_classification_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("clm_label_ids:\n{}".format(example["clm_labels"]))
        print("clm_labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["clm_labels"])), skip_special_tokens=False)
        ))
        print("cls_label_ids:\n{}".format(example["cls_labels"]))
        print("cls_labels:\n{}".format(
            [id2str[x] for x in example["cls_labels"] if x != IGNORE_INDEX]))


    if stage == "pt":
        preprocess_func = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        preprocess_func = preprocess_packed_supervised_dataset if data_args.sft_packing else preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        preprocess_func = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    elif stage == "lc":
        preprocess_func = preprocess_gennm_classification_dataset
        print_function = print_gennm_classification_example
    else:
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    if data_args.cache_path is not None and os.path.exists(data_args.cache_path):
        logger.warning("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(data_args.cache_path)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        if stage == "lc":
            column_names = ["query"]
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )
        # set mask_token, sep_token, cls_token, eos_token
        if tokenizer.mask_token is None:
            tokenizer._mask_token = tokenizer.unk_token
        if tokenizer.sep_token is None:
            tokenizer._sep_token = tokenizer.eos_token
        if tokenizer.cls_token is None:
            tokenizer._cls_token = tokenizer.bos_token



        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_file`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
