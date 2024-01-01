import torch
from torch import nn
from transformers import BertModel
from src.models.base_model import BaseModel
from typing import List


class WhereOperator(BaseModel):
    '''main network to predict the operator on a per predicate clause basis.'''
    MODEL_NAME = 'where_operator'

    def __init__(self, bert_model: BertModel,
                 num_operators: int,
                 operator_internal_size: int,
                 learning_rate: float,
                 bert_learning_rate: float,
                 point_accuracy: bool,
                 **kwargs
                 ):
        super(WhereOperator, self).__init__(
            learning_rate=learning_rate, bert_learning_rate=bert_learning_rate)
        self.hparams['model_name'] = WhereOperator.MODEL_NAME
        self.save_hyperparameters(ignore=list(kwargs.keys()) + ['bert_model'])
        self.bert_model: BertModel = bert_model
        self.operator_predictor_gru = nn.GRU(
            768, operator_internal_size, batch_first=True)
        self.operator_predictor: torch.nn.Module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(operator_internal_size, num_operators),
            nn.ReLU()
        )
        self.heads = [self.operator_predictor_gru, self.operator_predictor]
        self.point_accuracy = point_accuracy

    def forward(self, batch, use_gt=True, column_predictions: List[List[int]] = None):
        utterances, columns, trgs, _ = batch
        # B x n x 3 - n = number of clauses
        if use_gt:
            trgs = trgs['where']
        utterance_inputs = utterances['input_ids']
        cls, bert_output = self.bert_forward(utterances, columns)

        # Now have utterances and headers
        utterance_embeddings = bert_output[:, :utterance_inputs.shape[-1]]
        header_embeddings = bert_output[:, utterance_inputs.shape[-1]:]
        header_embeddings = self._extract_column_headers(
            header_embeddings, columns['pointers'])

        # Re-assemble input
        # (B * length of utterance + 1) x length of utterance + 1 x 768
        if use_gt:
            gru_inputs = [
                torch.cat((utterance_embeddings[i], header_embeddings[i, c[0]].unsqueeze(0))) for i in range(bert_output.shape[0]) for c in trgs[i]
            ]
        else:
            gru_inputs = [
                torch.cat((utterance_embeddings[i], header_embeddings[i, c].unsqueeze(0))) for i in range(bert_output.shape[0]) for c in column_predictions[i]
            ]
        if len(gru_inputs) == 0:
            return torch.Tensor([[0]]).to(self.device)

        gru_inputs = torch.stack(gru_inputs, dim=0).to(self.device)
        _, gru_context = self.operator_predictor_gru(gru_inputs)
        gru_context = gru_context.squeeze(0)
        # B_new x 4
        predictions = self.operator_predictor(gru_context)
        return predictions

    def loss(self, model_out, batch):
        # Pull out relevant data
        predictions = model_out
        if predictions.shape[0] == 1 and predictions.shape[1] == 1:
            _loss = self.loss_fn(predictions, torch.Tensor(
                [0]).long().to(self.device))
            _loss.requires_grad = True
            return _loss
        trgs = batch[2]['where']
        trgs = torch.Tensor([c[1] for x in trgs for c in x]
                            ).to(self.device).long()
        return self.loss_fn(predictions, trgs)

    def accuracy(self, model_out, batch):
        correctness_vector = self.accuracy_internal(model_out, batch)
        return correctness_vector.sum() / correctness_vector.shape[0]

    def accuracy_internal(self, model_out, batch):
        if model_out.shape[0] == 1 and model_out.shape[1] == 1:
            return torch.Tensor([True]).to(self.device)
        predictions = torch.argmax(model_out, 1)
        trgs = batch[2]['where']
        if not self.point_accuracy:
            trgs = torch.Tensor([c[1]
                                for x in trgs for c in x]).to(self.device)
        else:
            correctness_vector = []
            prediction_pointer = 0
            for clauses in trgs:
                correct = True
                for clause in clauses:
                    correct = correct and (
                        predictions[prediction_pointer] == clause[1])
                    prediction_pointer += 1
                correctness_vector.append(correct)
            correctness_vector = torch.Tensor(
                correctness_vector).bool().to(self.device)
            return correctness_vector
        return predictions == trgs

    def _get_batch_size(self, batch):
        trgs = batch[2]['where']
        trgs = torch.Tensor([c[1] for x in trgs for c in x]
                            ).to(self.device).long()
        batch_size = trgs.shape[0]
        del trgs
        if self.point_accuracy:
            return len(batch[2]['where'])
        else:
            return batch_size

    @ staticmethod
    def add_argparse_args(subparser):
        parser = subparser.add_parser(WhereOperator.MODEL_NAME)
        parser.add_argument('--multiheaded', action='store_true')
        parser.add_argument('--learning_rate', type=float, default=0.00005)
        parser.add_argument('--bert_learning_rate',
                            type=float, default=0.00005)
        parser.add_argument('--num_operators', type=int, default=4)
        parser.add_argument('--operator_internal_size', type=int, default=512)
        parser.add_argument('--point_accuracy', action='store_true')
