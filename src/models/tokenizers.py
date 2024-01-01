from transformers import BertTokenizer

"""Tokenizer definitions"""


def load_bert_tokenizer(tokenizer_name):
    if tokenizer_name == 'bert-base-uncased':
        return BertTokenizer.from_pretrained(
            'bert-base-uncased')
    
    assert False, f'Tokenizer could not be found: {tokenizer_name}'
