# Code of GenNm for review

We upload key implementations of GenNm for review.
We will publish all artifacts and data upon acceptance.

## Overview

This repository contains key components of GenNm.
Specifically, it includes: (1) the script uses to deduplicate datasets and identify data leakage, (2) the script used to process and generate the SymPO dataset, and (3) the script used by the name validation algorithm.

## Deduplication and Data Leakage Detection

The script is in `preprocess/dedup_dataset.py`.
It first uses function names to identify duplicate binaries (lines 33--59). Then it splits the dataset into training and testing sets, and uses string similarity to identify data leakage (lines 62--127).
To scale the deduplication to large dataset, given a function in test set, we first use its variable name list to identify potential leakage (lines 117--121). Then we only compute the full-body string similarity for training functions with high name list similarities (lines 123--127).

## SymPO Dataset Generation

Script `sympo/gen_sympo.py` is used to generate the SymPO dataset. It extracts data samples that the model does not perform well but has a good answer in top-K predictions (lines 69--270). Then statistic heuristics are used to filter out low-quality samples (lines 273--335).

## Name Validation Algorithm

The name validation algorithm is implemented in `name_validation/prop_names.py. The correlation extraction rules are implemented in the following functions:

```
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