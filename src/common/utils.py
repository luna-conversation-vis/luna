import json
import torch
from sys import stdout
from itertools import chain
from src.config import WhereCategories

"""
Helpful utility methods.
"""


def pprint_tree(tree):
    '''
    Pretty print a tree (obj class: dict).
    '''
    print(json.dumps(tree, indent=2, sort_keys=False))


def sql_state_tracking_json(split):
    '''
    Returns CoSQL SQL state tracking file after loading it in as a dict.
    '''
    fname = f'data/cosql_dataset/sql_state_tracking/cosql_{split}.json'
    with open(fname) as fp:
        return json.load(fp)


def table_metadata_json():
    '''
    Returns CoSQL table metadata.
    '''
    fname = 'data/cosql_dataset/tables.json'
    with open(fname) as fp:
        return json.load(fp)


def get_device():
    '''
    Returns current torch device.
    '''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_dict(d, file=stdout, prefix=''):
    '''
    Pretty print a dict.
    '''
    for k in d.keys():
        print(prefix, k + ':', d[k], file=file)


def collate_predictions(predictions):
    '''
    Given a set of predictions in dictionary form, collates them into one matrix per key.
    Input is dict: k -> list[Tensors,]
    '''
    new_preds = {}
    keys = predictions[0].keys()
    for k in keys:
        collated = [i[k] for i in predictions]
        if isinstance(collated[0], torch.Tensor) and len(collated[0].shape) > 0:
            new_preds[k] = torch.cat(collated)
        if isinstance(collated[0], torch.Tensor) and len(collated[0].shape) == 0:
            new_preds[k] = torch.Tensor([i.item() for i in collated])
        if isinstance(collated[0], list):
            new_preds[k] = list(chain(*collated))
    return new_preds


def accuracy_by_where_category(predictions):
    '''
    Returns accuracy broken down by where change prediction.
    '''
    correct = predictions['raw_accuracy']
    debug_info = predictions['debug_info']
    totals = {
        k: 0 for k in list(WhereCategories)
    }
    number_correct = {
        k: 0 for k in list(WhereCategories)
    }
    for was_correct, debug in zip(correct, debug_info):
        current_category = debug['where_categorization']
        totals[current_category] += 1
        if was_correct:
            number_correct[current_category] += 1
    return number_correct, totals


def accuracy_by_where_counts(predictions):
    '''
    Returns accuracy broken down by the number of predicates.
    '''
    correct = predictions['raw_accuracy']
    debug_info = predictions['debug_info']
    max_count = max([d['number_predicates'] for d in debug_info])
    number_correct_counts = [0 for _ in range(max_count+1)]
    total_counts = [0 for _ in range(max_count+1)]

    for was_correct, debug in zip(correct, debug_info):
        current_number_predicates = debug['number_predicates']
        if was_correct:
            number_correct_counts[current_number_predicates] += 1
        total_counts[current_number_predicates] += 1
    return number_correct_counts, total_counts


def _dict_to_device(d, device):
    '''
    Sends a dict of tensors to device
    '''
    for k in d.keys():
        d[k] = d[k].to(device)
    return d


def batch_to_device(batch, device):
    '''
    Sends a covis dataloader batch to device.
    '''
    return (_dict_to_device(batch[0], device), _dict_to_device(batch[1], device), batch[2], batch[3])


def str_is_num(s):
    '''
    Checks if s is a number in string form.
    '''
    if s is None:
        return False
    try:
        _t = float(s)
        return True
    except ValueError:
        return False
