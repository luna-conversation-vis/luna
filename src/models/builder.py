from src.models.where_operator import WhereOperator
from src.models.select_count import SelectCount
from src.models.select_col import SelectColumn
from src.models.where_count import WhereCount
from src.models.where_col import WhereColumn
from src.models.llm import load_bert
from typing import Dict


"""Model builders"""

_models = {
    'select_count': SelectCount,
    'select_col': SelectColumn,
    'where_count': WhereCount,
    'where_operator': WhereOperator,
    'where_col': WhereColumn
}

_tasks = {
    'select_count': 'select-count',
    'select_col': 'select-column',
    'where_count': 'predicate-count',
    'where_operator': 'predicate-operator',
    'where_col': 'predicate-column'
}


def get_model(args: Dict):
    model_class = _models[args.model_name]
    bert = load_bert()
    return model_class(bert_model=bert, **vars(args)), _tasks[args.model_name]


def add_subparsers(subparser):
    '''adds each model's args to the main argparser.'''
    for model_name in _models.keys():
        _models[model_name].add_argparse_args(subparser=subparser)


def load_checkpoint(model_name, path):
    model_class = _models[model_name]
    bert = load_bert()
    return model_class.load_from_checkpoint(path, bert_model=bert)
