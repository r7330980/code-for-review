import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer


class PreprocessLogitsForMetrics:

    def __call__(self, logits, labels):
        # argmax
        logits = torch.argmax(logits, dim=-1)        
        return logits


@dataclass
class GenNmClassificationMetrics:

    def __call__(self, eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        print("Preds shape", preds.shape)
        print("Labels shape", labels.shape)
        score_dict = {"acc": []}
        fout = open('dbg.txt', 'w')
        fout.write("Pred,Label\n")
        i = 0 
        for sample_pred, sample_label in zip(preds, labels):
            for pred, label in zip(sample_pred, sample_label):
                if label == IGNORE_INDEX:
                    continue
                score_dict["acc"].append(int(pred == label))
                if i < 10000:
                    i += 1
                    fout.write("%d,%d\n" % (pred, label))
        print("In total, we have %d samples, and %d samples are correct" % (len(score_dict["acc"]), sum(score_dict["acc"])))        
        return {k: float(np.mean(v)) for k, v in score_dict.items()}
