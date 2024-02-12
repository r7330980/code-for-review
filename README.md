# Code of GenNm for review

We upload our implementation of GenNm for review.
We will publish the artifacts and our dataset upon acceptance.

## Overview

GenNm leverages a generative model to predict variable names from decompiled
code of fully stripped binary programs.
The implementation consists of the following major components:

- A training pipeline that fine-tunes a stem model on decompiled code, a generative
model derived from the stem model, and a classification model derived from the
stem model.

- A inference pipeline that predicts variable names using the generative model
and the classification model, respectively. Then the predictions are combined
with our name validation algorithm developed based on program data flow and
name semantics.

## Implementation of Key Components

In this section, we list the implementation of the key components of GenNm.

### Training Pipeline

The training pipeline is under the `ft-models` directory.
We build our training pipeline based on a publicly available framework
[Llama-Factory](https://github.com/hiyouga/LLaMA-Factory).

We list three training scripts with all hyperparameters and configurations in the following files:

```
ft-models/train-stem-model.sh
ft-models/train-gen-model.sh
ft-models/train-cls-model.sh
```

For the generative model, we largely reuse the code in the Llama-Factory framework.
For the classification model, the definition of the model is in
`ft-models/src/gennm_model/gennm_modeling.py`.

At line 78, we can see that the hidden states output by the Transformer blocks
are forwarded to a classification layer which outputs the probability of each
candidate name:

```python
# transformer_outputs is the output of the transformer blocks
hidden_states = transformer_outputs[0]

class_logits = self.classifier(hidden_states)
```

To generate labels for the classification task,
we customize the data preprocessing function in `ft-models/src/llmtuner/dsets/preprocess.py`. Function `preprocess_gennm_classification_dataset` is corresponding to the preprocessing for the classification model.

At line 189, we identify from the encoded token IDs the positions corresponding to the variable names, and assign the corresponding class labels to the positions:

```python
# target_ids are the encoded token ids
for i, token_id in enumerate(target_ids):    
    # in our dataset, the variable names are organized in the json format
    # therefore, the beginning of a variable name always starts with ","
    # followed by "'".
    if token_id in interesting_ids:
        pattern_state = STATE_INTERESTED
    else:
        pattern_state = STATE_NO
    elif pattern_state == STATE_INTERESTED:
        next_answer = target_answer[answer_cnt]
        answer_cnt += 1
        if 'GENNM_SKIP' in next_answer:
            pass            
        # str2id is the vocab that maps the variable names to the class labels
        elif next_answer in str2id:
            # target_mask is the label for the classification task
            target_mask[i] = str2id[next_answer]
        else:
            target_mask[i] = str2id["<unk>"]
        pattern_state = STATE_NO
```


### Data Augmentation

The data augmentation is implemented in `ft-models/gennm_code/gen_hints.py`.
The file takes as input a mapping from identifiers to the names predicted by the models,
and the binary program (with the call graph).
It augments each function with the context information, and outputs
the augmented data in a `jsonl` file.

Specifically, the code logic for augmentation is implemented in `process_one_binary` function.
Given a function, it first extracts the callers and callees:

```python
callers = set()
callees = set()
for caller, callee in call_graph.edges:
    if callee == func_name:
        callers.add(caller)
    if caller == func_name:
        callees.add(callee)
```

Then it collect the predicted names in the callers and callees:

```python
for caller in callers:
    if (binary.prog_name, caller) in prog_func2entry:
        caller_hints.extend(prog_func2entry[(binary.prog_name, caller)])
for callee in callees:
    if (binary.prog_name, callee) in prog_func2entry:
        callee_hints.extend(prog_func2entry[(binary.prog_name, callee)])
```

### Inference Pipeline

The code to query the generative and classification models are implmeneted in
```
inferece/scripts/infer_gen_model.py
inferece/scripts/infer_cls_model.py
```

The query to the generative model is standard.
We simply encode the query function, and let the model generate the top-3 answers:

```python
ret = model.generate(
    input_ids=ask_tokenized,
    max_new_tokens=max_new_token_adjusted,
    temperature=self.config.temperature,
    num_return_sequences=3,
    # ...
)
```

For the query to the classification model, we manually append the previously
predicted names to the input tokens, and query the model to get the probability
of each candidate name:

```python
while asked_cnt < len(asked_vars):   
    var_to_ask = asked_vars[asked_cnt]
    answer_str = "{"
    for var_id, answer_id in answers:
        answer_str += "'%s': '%s', " % (var_id, answer_id)
    answer_str += "'%s': '" % var_to_ask
    # here we update the input tokens with the previously predicted names
    current_ask_str = all_text + answer_str
    tokenized = tokenizer(current_ask_str, return_tensors="pt")['input_ids'].cuda()
    output = model(tokenized)
    logits = output.class_logits[0][-1]
    preds = torch.softmax(logits, dim=-1)
    answers_logits.append(logits)
```

### Name Validation

The name validation algorithm is implemented in `inference/scripts/prop_names.py`.
The correlation extraction rules are implemented in the following functions:
```python
# call
_from_callee_args
# ret
_from_callee_return
# call-R
_from_caller_args
# ret-R
_from_caller_return
# assign and assign-R
_among_direct_use
```

The semantics voting algorithm is implemented in the following code snippet:
```python
with torch.no_grad():
    prop_name_embs = [
        torch.mean(embeddings(name_token), dim=0)
        for name_token in prop_name_tokens
    ]
    prop_name_embs_tensor = torch.stack(prop_name_embs)
    # ... normalization
    similarity = torch.matmul(
        name_embs_tensor, prop_name_embs_tensor.transpose(0, 1)
    )
    scores = torch.mean(similarity, dim=-1)
    # find the name that is most similar to all candidates
    sorted_scores = torch.argsort(scores, descending=True)
    selected = name2name_list_entry[name_candidates[sorted_scores[0]]]
    # `k` stores the (fid, vid) corr. to a variable
    new_name_selections[k] = selected
```

### Parsing the Decompiled Code

Before running our name validation algorithm, we parse the decompiled code to 
a simple language. The parsing code is implemented in `clang-parser/`.
It is built on top of the clang AST parser.

The core logic for building the simple language is in `clang-parser/analysis/ir_builder.cc`.

### Evaluation Metrics

The evaluation metrics are implemented in `eval/eval_utils.py`.
Our adapted metrics and the original metrics in SymLM are implemented in
`score_name` and `score_name_ori`, respectively.

