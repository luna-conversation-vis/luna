from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as nnF
from copy import deepcopy

"""
Various collation functions to provide to the dataloader. We don't use a standard tensor load from the dataloader so we need to provide a collation method.
"""


def _pad_tensors(tensors: List[torch.Tensor], length=-1, padding_value=0., squeeze=False):
    '''
    Pads a list of tensors to the largest size.
    '''

    deepcopy_tensors = deepcopy(tensors)

    if squeeze:
        new_tensors = []
        for t in tensors:
            if len(t.shape) > 1:
                new_tensors.append(t[0])
            else:
                new_tensors.append(t)
        tensors = new_tensors
    if length == -1:
        return pad_sequence(tensors,
                            batch_first=True,
                            padding_value=padding_value)
    else:
        # it is very dangerous to change the original input data!
        # change tensors to deepcopy_tensors
        first_tensor = deepcopy_tensors[0]
        first_tensor = nnF.pad(first_tensor,
                               (0, length - first_tensor.shape[-1]),
                               value=padding_value)
        deepcopy_tensors[0] = first_tensor
        return pad_sequence(deepcopy_tensors,
                            batch_first=True,
                            padding_value=padding_value)

def collate(batch):
    '''
    Main collate function. Collates a batch.
    '''
    debug_info = [b['DEBUG'] for b in batch]
    utterances = [b['utterance'] for b in batch]
    columns = [b['columns'] for b in batch]
    utterances_collated, columns_collated = {}, {}

    # Collate bert inputs
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        for uncollated, collated in zip([utterances, columns],
                                        [utterances_collated,
                                         columns_collated]):
            curr_tensors = [i[key] for i in uncollated]
            collated[key] = _pad_tensors(curr_tensors)

    # Collate column pointers
    column_input_ids = columns_collated['input_ids']
    columns_collated['pointers'] = _pad_tensors(
        [i['pointers'] for i in columns],
        padding_value=column_input_ids.shape[-1] - 1).long()

    # Collate the targets
    targets = {}

    select_targets = [torch.Tensor(b['select']).long() for b in batch]

    targets['select'] = select_targets

    # Pad the selected columns to the same shape as input columns
    targets['select_pad'] = _pad_tensors(select_targets, length = columns_collated['pointers'].shape[1], padding_value = -1)

    # Transform selected columns to one-hot
    select_targets_onehot = torch.zeros_like(targets['select_pad'])

    for rid, row in enumerate(targets['select_pad'].cpu().tolist()):
        for cid, col in enumerate(row):
            if col >= 0:
                select_targets_onehot[rid][col] = 1

    targets['select_onehot'] = select_targets_onehot.float()

    targets['where'] = [b['where'] for b in batch]

    return (utterances_collated, columns_collated, targets, debug_info)

def inference_collate(batch):
    '''
    Main collate function. Collates a batch. 
    '''
    # debug_info = [b['DEBUG'] for b in batch]
    debug_info = []
    utterances = [b['utterance'] for b in batch]
    columns = [b['columns'] for b in batch]
    utterances_collated, columns_collated = {}, {}

    # Collate bert inputs
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        for uncollated, collated in zip([utterances, columns],
                                        [utterances_collated,
                                         columns_collated]):
            curr_tensors = [i[key] for i in uncollated]
            collated[key] = _pad_tensors(curr_tensors)

    # Collate column pointers
    column_input_ids = columns_collated['input_ids']
    columns_collated['pointers'] = _pad_tensors(
        [i['pointers'] for i in columns],
        padding_value=column_input_ids.shape[-1] - 1).long()

    # Collate the targets
    targets = {}

    return (utterances_collated, columns_collated, targets, debug_info)
