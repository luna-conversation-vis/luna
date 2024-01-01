from typing import Dict, List
import torch
from src.models.tokenizers import load_bert_tokenizer

"""
Set of transforms for dataloading.
"""


# Special tokens in BERT's tokenizer
BERT_SPECIAL_TOKENS = {
    ',': 1010,
    '|': 1064,
    ':': 1024
}


def default_column_transforms():
    '''Helper method for the default transforms'''
    return [
        CleanColumnsTransform,
        CleanStringTransform,
        TokenizeTransform(),
        DictToTensorTransform,
        ExtractColumnPointersTransform,
    ]


def default_utterance_transforms():
    '''Helper method for the default transforms'''
    return [
        CleanStringTransform,
        TokenizeTransform(),
        DictToTensorTransform,
    ]


def default_where_transform():
    '''Helper method for the default transforms'''
    return [
        PredicateOperationEncodeTransform
    ]


def ToLongTransform(X):
    return X.long()


def PrependSOSToken(clauses):
    for i in range(len(clauses)):
        clauses[i].insert(0, '[SOS]')
    return clauses


def DictToTensorTransform(dict: Dict):
    for k in dict.keys():
        dict[k] = torch.Tensor(dict[k]).long()
    return dict


class TokenizeTransform():
    '''Simple transform to run tokenization.'''

    def __init__(self, tokenizer_name='bert-base-uncased', **kwargs):
        self.tokenizer = load_bert_tokenizer(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.kwargs = kwargs

    def __call__(self, X: str):
        return self.tokenizer(X, **self.kwargs)


class TokenizeClausesTransform():
    '''Simple transform to run tokenization on where clauses.'''

    def __init__(self, tokenizer_name='bert-base-uncased-lstm'):
        self.tokenizer = load_bert_tokenizer(tokenizer_name)
        self.tokenizer_name = tokenizer_name

    def __call__(self, clauses: str):
        return self.tokenizer(clauses, return_tensors='pt')


def CleanStringTransform(X: str):
    return X.strip()


def ExtractColumnPointersTransform(tokenizer_output):
    '''returns pointers to tokens representing columns'''
    pointers = []
    input_ids = tokenizer_output['input_ids']
    for idx, tok in enumerate(input_ids):
        # Middle and last tokens
        # 102 -> [SEP]
        if tok == 102 and idx < input_ids.shape[0] - 1:
            pointers.append(idx + 1)
        # First header
        # 101 -> [CLS]
        elif tok == 101 and idx == 0:
            pointers.append(idx + 1)
    pointers = torch.Tensor(pointers).long()
    tokenizer_output['pointers'] = pointers
    return tokenizer_output


def CleanColumnsTransform(columns: List[str]):
    columns = [i.strip() for i in columns]
    columns_joined = ' [SEP] '.join(columns)
    return columns_joined


def PredicateOperationEncodeTransform(clauses):
    ops = ['eq', 'gt', 'lt', 'neq']
    new_clauses = []
    for clause in clauses:
        assert clause[1] in ops, f'Found an op not in ops: {clause}'
        current_new_clause = [clause[0], ops.index(clause[1]), clause[2]]
        new_clauses.append(current_new_clause)
    return new_clauses


def PrepareClausesForTokenization(clauses):
    for i in range(len(clauses)):
        col, op, val = clauses[i]
        col = f'column{col}'
        op = f'{op}operator'
        clauses[i] = [col, op, val]
    return clauses


def ToTensorTransform(X):
    if len(X) == 0:
        return torch.Tensor([0])
    return torch.Tensor(X)


def ConsolidateLists(clauses):
    clauses = [[str(c) for c in clause]for clause in clauses]
    new_clauses = [" ".join(clause) for clause in clauses]
    return " ".join(new_clauses)


class TokenizeBatchTransform:
    '''tokenization on a batch'''

    def __init__(self, tokenizer_name='bert-base-uncased-lstm'):
        self.tokenizer = load_bert_tokenizer(tokenizer_name)
        self.tokenizer_name = tokenizer_name

    def __call__(self, strings: List[str]):
        tokenized = self.tokenizer(
            strings, return_tensors='pt', add_special_tokens=False)
        return tokenized