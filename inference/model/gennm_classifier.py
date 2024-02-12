import torch
import torch.nn as nn
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.utils import ModelOutput
from typing import Optional, List
from dataclasses import dataclass

IGNORE_INDEX = -100

@dataclass
class GenNmDecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    clm_loss: Optional[torch.FloatTensor] = None
    clm_logits: Optional[torch.FloatTensor] = None
    class_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[List[torch.FloatTensor]] = None
    

class GenNmDecoder(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_classes = config.num_classes
        self.hidden_size = config.hidden_size
        # if mlp_cls does not in config
        if not hasattr(config, "mlp_cls"):
            config.mlp_cls = False
        self.mlp_cls = config.mlp_cls
        if config.mlp_cls:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_classes, bias=False)
            )
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_classes, bias=False)
        self.cls_loss_weight = config.cls_loss_weight
        self.no_clm = config.no_clm
        self.id2str = config.id2str
        self.str2id = config.str2id
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        clm_labels: Optional[torch.LongTensor] = None,
        cls_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        class_logits = self.classifier(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     # from pytorch document for torch.max
        #     # "If there are multiple maximal values in a reduced row then 
        #     # the indices of the first maximal value are returned."
        #     if input_ids is not None:
        #         sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
        #             logits.device
        #         )
        #     else:
        #         sequence_lengths = -1
    
        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)        
        if clm_labels is not None and not self.no_clm:
            clm_logits = self.lm_head(hidden_states)
            shift_logits = clm_logits[..., :-1, :].contiguous()
            shift_labels = clm_labels[..., 1:].contiguous().to(shift_logits.device)
            clm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            clm_loss = torch.tensor(0.0).to(class_logits.device)
            clm_logits = None

        if cls_labels is not None:
            shift_class_logits = class_logits[..., :-1, :].contiguous()
            shift_class_labels = cls_labels[..., 1:].contiguous().to(shift_class_logits.device)
            class_loss = loss_fct(shift_class_logits.view(-1, shift_class_logits.size(-1)), shift_class_labels.view(-1))
        else:
            class_loss = torch.tensor(0.0).to(class_logits.device)
        if not self.no_clm:
            loss = clm_loss + self.cls_loss_weight * class_loss
        else:
            loss = self.cls_loss_weight * class_loss
        # loss = clm_loss + self.cls_loss_weight * class_loss
            
        return GenNmDecoderOutput(
            loss=loss,
            cls_loss=class_loss,
            clm_loss=clm_loss,
            clm_logits=clm_logits,
            class_logits=class_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )
    