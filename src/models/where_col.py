from numpy import number
import torch
from transformers import BertModel
from src.models.base_model import BaseModel
from src.modules.column_attention import ColumnAttention


class WhereColumn(BaseModel):
    '''base network to predict the relevant column for a where / predicate clause'''
    MODEL_NAME = 'where_col'

    def __init__(self, bert_model: BertModel,
                 multiheaded: bool,
                 learning_rate: float,
                 bert_learning_rate: float,
                 num_CA_heads: int,
                 **kwargs
                 ):
        super(WhereColumn, self).__init__(
            learning_rate=learning_rate, bert_learning_rate=bert_learning_rate)
        self.save_hyperparameters(ignore=list(kwargs.keys()) + ['bert_model'])
        self.bert_model: BertModel = bert_model
        self.col_attn: ColumnAttention = ColumnAttention(
            multiheaded=multiheaded, num_heads=num_CA_heads)
        self.heads = [self.col_attn]

    def forward(self, batch):
        utterances, columns, _, _ = batch
        utterance_inputs = utterances['input_ids']

        _, bert_output = self.bert_forward(utterances, columns)

        # Predict the relevant columns
        utterance_embeddings = bert_output[:, :utterance_inputs.shape[-1]]
        header_embeddings = bert_output[:, utterance_inputs.shape[-1]:]
        # B x max(|Cols_i|)
        header_embeddings = self._extract_column_headers(
            header_embeddings, columns['pointers'])
        # B x max(|Cols_i|)
        col_attn_out = self.col_attn(utterance_embeddings, header_embeddings)

        return col_attn_out

    def loss(self, model_out, batch):
        # Pull out relevant data
        where_targets = batch[2]['where']
        column_probabilities = model_out
        columns_gt = [[j[0] for j in i] for i in where_targets]
        max_num_cols = column_probabilities.shape[-1]
        target = []
        for trg in columns_gt:
            target.append(self.one_hot_encoding(trg, max_num_cols))
        target = torch.Tensor(target).to(self.device)
        return self.loss_fn(column_probabilities, target)

    def accuracy_internal(self, model_out, batch):
        correctness_vector = []
        where_targets = batch[2]['where']
        for index in range(model_out.shape[0]):
            correct = True
            trgs = set([i[0] for i in where_targets[index]])
            number_predicates = len(trgs)
            if number_predicates > 0:
                predictions = set(torch.topk(
                    model_out[index], largest=True, k=number_predicates).indices.tolist())
                if predictions != trgs:
                    correct = False
            correctness_vector.append(correct)
        return torch.Tensor(correctness_vector).to(self.device)

    def accuracy(self, model_out, batch):
        correctness_vector = self.accuracy_internal(model_out, batch)
        return correctness_vector.sum() / self._get_batch_size(batch)

    def one_hot_encoding(self, targets, max_length):
        new_target = [0 for _ in range(max_length)]
        for i in targets:
            new_target[i] = 1
        return new_target

    @ staticmethod
    def add_argparse_args(subparser):
        parser = subparser.add_parser(WhereColumn.MODEL_NAME)
        parser.add_argument('--multiheaded', action='store_false')
        parser.add_argument('--learning_rate', type=float, default=0.00005)
        parser.add_argument('--bert_learning_rate',
                            type=float, default=0.00005)
        parser.add_argument('--num_CA_heads', type=int, default=8)
