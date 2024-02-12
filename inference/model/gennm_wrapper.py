import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import gennm_utils


class GenNmConfig:
    """
    This class specifies how we instantiate a GenNmWrapper model.
    It records information about 
    - an original function string,
    - a map for variables from current name to target name
    - a map for variables from current name to candidate names  
    """

    def __init__(self, func_str, id_maps, candi_maps, candi_length):
        self.func_str = func_str
        self.id_maps = id_maps
        self.candi_maps = candi_maps
        self.candi_length = candi_length
        self.gt = {}
        for k, v in id_maps.items():
            self.gt[k] = candi_maps[k].index(v)


class GenNmWrapper(nn.Module):
    def __init__(self, llama, tokenizer, gennm_config,
                 temperature=0.01,
                 one_hot=False) -> None:
        super().__init__()
        self.llama = llama
        self.tokenizer = tokenizer
        self.gennm_config = gennm_config
        self.temperature = temperature
        self.one_hot = one_hot
        self._initialize_gennm_model()

    def _encode_and_pad_candi_maps(self):
        """
        align candidate names to a same length
          if a candi name is longer, we truncate it
          if a candi name is shorter, we pad it with white space
        """
        candi_length = self.gennm_config.candi_length
        candi_maps = self.gennm_config.candi_maps
        tokenizer = self.tokenizer

        name_length = candi_length
        # white_space = tokenizer.encode(' ')[1]
        white_space = 29871
        encoded_candi_maps = {}
        encoded_candi_maps_with_previous = {}
        for id_name in candi_maps.keys():
            cands = candi_maps[id_name]
            encoded_candi_maps[id_name] = []
            encoded_candi_maps_with_previous[id_name] = []
            for candi in cands:
                encoded = tokenizer.encode(candi, return_tensors="pt")[0, 1:]
                if len(encoded) < name_length:
                    encoded = torch.cat([encoded, torch.ones(
                        name_length - len(encoded)).long() * white_space], dim=0)
                encoded = encoded[:name_length]
                encoded_candi_maps[id_name].append(encoded)

                # pretend that I am adjacent to the previous token
                encoded_w_prev = tokenizer.encode(","+candi, return_tensors="pt")[0][2:]
                if len(encoded_w_prev) < name_length:
                    encoded_w_prev = torch.cat([encoded_w_prev, torch.ones(
                        name_length - len(encoded_w_prev)).long() * white_space], dim=0)
                encoded_w_prev = encoded_w_prev[:name_length]
                encoded_candi_maps_with_previous[id_name].append(encoded_w_prev)

        return encoded_candi_maps, encoded_candi_maps_with_previous

    def _embed_tokens(self, ids_tensor):
        return self.llama.model.embed_tokens(ids_tensor)

    def _encode_func_str_with_slices(self):
        func_str_ori = self.gennm_config.func_str
        tokenizer = self.tokenizer
        candi_length = self.gennm_config.candi_length
        id_maps = self.gennm_config.id_maps

        id_slices = {}
        for k, v in id_maps.items():
            id_slices[k] = gennm_utils.find_variable_indices(func_str_ori, k)

        # encode the original function string
        slices_all_in_one = {}
        for var_id, slices in id_slices.items():
            for slice_instance in slices:
                slices_all_in_one[slice_instance] = var_id

        slices_sorted = sorted(slices_all_in_one.items())
        # input segments: list of tuples:
        # (input seg, the related var after the seg)
        input_segments = []
        last_idx = 0        
        for current_slice in slices_sorted:
            pos = current_slice[0]
            related_var = current_slice[1]
            input_segments.append((func_str_ori[last_idx:pos[0]], related_var))
            last_idx = pos[1]
        input_segments.append((func_str_ori[last_idx:], None))

        place_holder = "?"
        encoded_segments = []
        encoded_slice_indices = {}
        current_idx = 0
        self.encoded_pos_has_leading_space = {}
        prev_has_tailing_space = False        
        for seg, related_var in input_segments:
            # may having trailing white space, remove
            if seg[-1] == ' ':
                seg = seg[:-1]
                prev_has_tailing_space = True
            else:
                prev_has_tailing_space = False
            if seg[0] == ' ':
                seg = seg[1:]
                i_have_leading_space = True
            else:
                i_have_leading_space = False
            # encode the current segment, skip <s>
            if i_have_leading_space:
                # meaning that I am not a part of the previous token
                current_seg_encoded = tokenizer.encode(
                    seg, return_tensors="pt")[0, 1:]
            else:
                # whether seg[0] is in 0-9a-zA-Z
                if seg[0].isalnum():
                    fake_prefix=","
                else:
                    fake_prefix="a"                    
                current_seg_encoded = tokenizer.encode(
                    fake_prefix+seg, return_tensors="pt")[0][2:]
            encoded_segments.append(current_seg_encoded)
            # update the current index in the encoded string
            current_idx += len(current_seg_encoded)
            # if the variable related to the current segment is not None
            # append a sequence of place holders,
            # and record the position of the place holders
            if related_var is not None:
                encoded_replacement = tokenizer.encode(
                    ' '.join([place_holder] * candi_length),
                    return_tensors="pt")[0, 1:]
                if related_var not in encoded_slice_indices:
                    encoded_slice_indices[related_var] = []
                encoded_slice_indices[related_var].append(
                    (current_idx, current_idx + candi_length))
                encoded_segments.append(encoded_replacement)
                if prev_has_tailing_space:
                    self.encoded_pos_has_leading_space[current_idx] = True
                else:
                    self.encoded_pos_has_leading_space[current_idx] = False
                current_idx += len(encoded_replacement)

        encoded_ret = torch.cat(encoded_segments, dim=0)
        return encoded_ret, encoded_slice_indices

    def _create_name_selection_vectors(self):
        name_selection_vectors = {}
        for var_id, candidates in self.encoded_candi_maps.items():
            num_candidates = len(candidates)
            name_selection_vectors[var_id] = torch.zeros(
                (1, num_candidates), dtype=self.llama.dtype)
            if self.one_hot:
                # # randomly select one candidate
                # selected_idx = torch.randint(
                #     0, num_candidates, (1,), dtype=torch.long)
                # name_selection_vectors[var_id][0, selected_idx] = 1

                # encode var_id
                if var_id in self.gennm_config.candi_maps[var_id]:
                    # get idx
                    idx = self.gennm_config.candi_maps[var_id].index(var_id)
                    name_selection_vectors[var_id][0, idx] = 1
                else:
                    selected_idx = torch.randint(
                        0, num_candidates, (1,), dtype=torch.long)
                    print("I selected %d for %s" %(selected_idx, var_id))
                    name_selection_vectors[var_id][0, selected_idx] = 1                    
            else:
                # uniformly initialize all candidates
                name_selection_vectors[var_id][0, :] = 1/num_candidates
                

        return name_selection_vectors

    def _initialize_gennm_model(self):
        self.encoded_candi_maps, self.encoded_candi_maps_w_prev = self._encode_and_pad_candi_maps()        

        candi_maps_embs = {}
        for var_id, candidates in self.encoded_candi_maps.items():
            candi_maps_embs[var_id] = self._embed_tokens(
                torch.stack(candidates, dim=0).to(self.llama.device))            
        self.candi_maps_embs = candi_maps_embs

        candi_maps_embs_w_prev = {}
        for var_id, candidates in self.encoded_candi_maps_w_prev.items():
            candi_maps_embs_w_prev[var_id] = self._embed_tokens(
                torch.stack(candidates, dim=0).to(self.llama.device))
        self.candi_maps_embs_w_prev = candi_maps_embs_w_prev

        self.name_selection_vectors = self._create_name_selection_vectors()

        encoded_func_str, encoded_slice_indices = self._encode_func_str_with_slices()
        self.func_str_emb = self._embed_tokens(encoded_func_str.cuda())
        self.encoded_func_str = encoded_func_str
        self.encoded_slice_indices = encoded_slice_indices

        # use module dict to register the parameters
        self.name_selection_vectors = nn.ParameterDict(
            self.name_selection_vectors)
        self.func_str_emb = self.func_str_emb.detach()
        for var_id, _ in self.candi_maps_embs.items():
            self.candi_maps_embs[var_id] = self.candi_maps_embs[var_id].detach(
            )
        self.name_selection_vectors.requires_grad_(True)
        # name selection vectors require gradients, but llama does not
        self.llama.requires_grad_(False)

    def _get_embedding(self):
        current_round_emb = self.func_str_emb.clone()
        for var_id, slices in self.encoded_slice_indices.items():
            current_var_emb = self.candi_maps_embs[var_id]            
            current_var_emb_w_prev = self.candi_maps_embs_w_prev[var_id]
            current_var_name_selection = self.name_selection_vectors[var_id]
            current_var_name_selection_softmax = torch.softmax(
                current_var_name_selection/self.temperature, dim=1)
            num_candi, candi_len, emb_dim = current_var_emb.shape            
            selected_emb = current_var_name_selection_softmax @ current_var_emb.view(
                num_candi, -1)
            selected_emb_w_prev = current_var_name_selection_softmax @ current_var_emb_w_prev.view(
                num_candi, -1)
            selected_emb_reshaped = selected_emb.view(candi_len, emb_dim)
            selected_emb_w_prev_reshaped = selected_emb_w_prev.view(candi_len, emb_dim)
            for slice_instance in slices:
                if self.encoded_pos_has_leading_space[slice_instance[0]]:                    
                    current_round_emb[slice_instance[0]:slice_instance[1]] = selected_emb_reshaped
                else:
                    current_round_emb[slice_instance[0]:slice_instance[1]] = selected_emb_w_prev_reshaped

        bos_emb = self._embed_tokens(torch.tensor(
            [self.tokenizer.bos_token_id]).cuda())
        ret_emb = torch.cat([bos_emb, current_round_emb], dim=0)
        return ret_emb

    def _get_target_id(self):
        ret = self.encoded_func_str.clone()
        for var_id, slices in self.encoded_slice_indices.items():
            current_var_name_selection = self.name_selection_vectors[var_id]
            current_var_candidate_ids = self.encoded_candi_maps[var_id]
            current_var_candidate_ids_w_prev = self.encoded_candi_maps_w_prev[var_id]
            # # argmax
            # selected_idx = torch.argmax(current_var_name_selection)

            # sample the selected idx according to probability distribution
            # create a probability distribution from the name selection vector
            minimal_value = torch.min(current_var_name_selection)            
            if minimal_value < 0:
                prob_dist = current_var_name_selection - minimal_value
            else:
                prob_dist = current_var_name_selection
            selected_idx = torch.multinomial(
                prob_dist, 1).squeeze(0)
                        
            # # randomly select one candidate
            # selected_idx = torch.randint(
            #     0, len(current_var_name_selection[0]), (1,), dtype=torch.long)

            # # dbg
            # selected_idx = 443
            
            selected_ids = current_var_candidate_ids[selected_idx]
            selected_ids_w_prev = current_var_candidate_ids_w_prev[selected_idx]
            for slice_instance in slices: 
                if self.encoded_pos_has_leading_space[slice_instance[0]]:
                    ret[slice_instance[0]:slice_instance[1]] = selected_ids
                else:
                    ret[slice_instance[0]:slice_instance[1]] = selected_ids_w_prev
        ret = torch.cat(
            [torch.tensor([self.tokenizer.bos_token_id]), ret], dim=0).cuda()
        return ret

    def forward(self, input_ids=None, dbg=False, **kwargs):
        # # I don't need input
        
        # # print(self.name_selection_vectors['a1'])
        # # print(self.name_selection_vectors['a1'].grad)
        # for k, v in self.name_selection_vectors.items():
        #     # pass
        #     # normalize
        #     self.name_selection_vectors[k] = torch.softmax(
        #         self.name_selection_vectors[k]/self.temperature, dim=1)

        emb = self._get_embedding()
        ids = self._get_target_id()
        # if 'cnt' not in self.__dict__:
        #     self.cnt = 0
        # self.cnt += 1
        # if self.cnt < 10:
        #     print(tokenizer.decode(ids[:100]))        
        ret = self.llama(inputs_embeds=emb[:1024].unsqueeze(0),
                         labels=ids[:1024].unsqueeze(0))
        if dbg:
            return ret, emb, ids
        return ret

    def get_gennm_weights(self):
        ret = {}
        ret['name_selection_vectors'] = self.name_selection_vectors
        return ret
