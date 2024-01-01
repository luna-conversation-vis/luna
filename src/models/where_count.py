import torch
from torchmetrics import Accuracy
from transformers import BertModel
from src.models.base_model import BaseModel
from src.modules.affine_classifier import ClassificationHead
from torch.nn.functional import pad


class WhereCount(BaseModel):
    '''network to predict the number of predicates. Formulated as a classification task'''
    MODEL_NAME = 'where_count'

    def __init__(self, bert_model: BertModel,
                 max_op_count: int,
                 op_internal_size: int,
                 learning_rate: float,
                 bert_learning_rate: float,
                 **kwargs
                 ):
        super(WhereCount, self).__init__(
            learning_rate=learning_rate, bert_learning_rate=bert_learning_rate)
        self.hparams['model_name'] = WhereCount.MODEL_NAME
        self.save_hyperparameters(ignore=list(kwargs.keys()) + ['bert_model'])
        self.bert_model: BertModel = bert_model
        self.where_count_classifier: ClassificationHead = ClassificationHead(
            hidden_state_size=op_internal_size, n_classes=max_op_count)
        self.heads = [self.where_count_classifier]

    def forward(self, batch):
        utterances, columns, _, _ = batch
        cls, _ = self.bert_forward(utterances, columns)

        # B x self.max_op_count
        predicated_number_predicates = self.where_count_classifier(cls)
        a = predicated_number_predicates

        if a.shape[-1] < 5:
            curr_num_classes = a.shape[-1]
            predicated_number_predicates = pad(predicated_number_predicates, [
                                               0, 5-curr_num_classes], value=0.)

        return predicated_number_predicates

    def loss(self, model_out, batch):
        # Pull out relevant data
        current_device = next(self.parameters()).device
        where_targets = batch[2]['where']
        expected_predicate_counts = torch.Tensor(
            [len(i) for i in where_targets]).long().to(current_device)
        predicated_number_predicates = model_out
        return self.loss_fn(predicated_number_predicates,
                            expected_predicate_counts)

    def accuracy_internal(self, model_out, batch):
        current_device = self.device
        where_targets = batch[2]['where']
        expected_predicate_counts = torch.Tensor(
            [len(i) for i in where_targets]).long().to(current_device)
        predicated_number_predicates = model_out
        return torch.argmax(predicated_number_predicates, dim=1) == expected_predicate_counts

    def accuracy(self, model_out, batch):
        predictions_scores = self.accuracy_internal(
            model_out, batch)
        return predictions_scores.sum() / self._get_batch_size(batch)

    @ staticmethod
    def add_argparse_args(subparser):
        parser = subparser.add_parser(WhereCount.MODEL_NAME)
        parser.add_argument('--learning_rate', type=float, default=0.00005)
        parser.add_argument('--bert_learning_rate',
                            type=float, default=0.00005)
        parser.add_argument('--max_op_count', type=int, default=5)
        parser.add_argument('--op_internal_size', type=int, default=1028)
