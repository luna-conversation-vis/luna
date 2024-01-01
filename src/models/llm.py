from transformers import BertModel

"""Load large language models"""


def load_bert():
    return BertModel.from_pretrained('bert-base-uncased',
                                     output_hidden_states=False)
