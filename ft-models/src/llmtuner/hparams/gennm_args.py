from typing import List, Literal, Optional
from dataclasses import dataclass, field
import json

@dataclass
class GenNmArguments:
    class_num: Optional[int] = field(
        default=23,
        metadata={"help": "The number of classes to classify."}
    )

    class_id_to_str: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing class id to string mapping."}
    )

    freeze_llm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM."}
    )

    freeze_layer_num: Optional[int] = field(
        default=-1,
        metadata={"help": "The number of layers to freeze."}
    )

    cls_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of classification loss."}
    )    

    no_clm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use CLM."}
    )

    mlp_cls: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use MLP classifier."}
    )

    def __post_init__(self):
        if self.class_id_to_str is not None:
            with open(self.class_id_to_str, "r") as f:
                self.class_id_to_str = json.load(f)
            id2str = {}
            for k,v in self.class_id_to_str.items():
                id2str[int(k)] = v
            self.class_id_to_str = id2str
            self.class_num = len(self.class_id_to_str)