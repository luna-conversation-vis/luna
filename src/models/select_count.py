import torch
from torchmetrics import Accuracy
from transformers import BertModel
from src.models.base_model import BaseModel
from src.modules.affine_classifier import ClassificationHead
from torch.nn.functional import pad

MAX_SELECT_COUNT = 6


class SelectCount(BaseModel):
    '''network to predict the number of selected attributes. Formulated as a classification task'''
    MODEL_NAME = 'select_count'

    def __init__(self, bert_model: BertModel,
                 op_internal_size: int,
                 learning_rate: float,
                 bert_learning_rate: float,
                 max_attribute_count: int,
                 **kwargs
                 ):
        super(SelectCount, self).__init__(
            learning_rate=learning_rate, bert_learning_rate=bert_learning_rate)
        self.hparams['model_name'] = SelectCount.MODEL_NAME
        self.save_hyperparameters(ignore=list(kwargs.keys()) + ['bert_model'])
        self.bert_model: BertModel = bert_model
        self.select_count_classifier: ClassificationHead = ClassificationHead(
            hidden_state_size=op_internal_size, n_classes=max_attribute_count + 1)
        self.heads = [self.select_count_classifier]

    def forward(self, batch):
        utterances, columns, _, _ = batch
        cls, _ = self.bert_forward(utterances, columns)

        # B x self.max_attribute_count
        predicated_number_select = self.select_count_classifier(cls)
        a = predicated_number_select

        if a.shape[-1] < 5:
            curr_num_classes = a.shape[-1]
            predicated_number_select = pad(predicated_number_select, [
                                               0, 5-curr_num_classes], value=0.)

        return predicated_number_select

    def loss(self, model_out, batch):
        # Pull out relevant data
        current_device = next(self.parameters()).device
        select_targets = batch[2]['select']
        expected_select_counts = torch.Tensor(
            [len(i) for i in select_targets]).long().to(current_device)
        predicated_number_select = model_out
        return self.loss_fn(predicated_number_select,
                            expected_select_counts)

    def accuracy_internal(self, model_out, batch):
        current_device = self.device
        select_targets = batch[2]['select']
        expected_select_counts = torch.Tensor(
            [len(i) for i in select_targets]).long().to(current_device)
        predicated_number_select = model_out
        return torch.argmax(predicated_number_select, dim=1) == expected_select_counts

    def accuracy(self, model_out, batch):
        predictions_scores = self.accuracy_internal(
            model_out, batch)
        return predictions_scores.sum() / self._get_batch_size(batch)

    @ staticmethod
    def add_argparse_args(subparser):
        parser = subparser.add_parser(SelectCount.MODEL_NAME)
        parser.add_argument('--learning_rate', type=float, default=0.00005)
        parser.add_argument('--bert_learning_rate',
                            type=float, default=0.00005)
        parser.add_argument('--max_attribute_count', type=int, default=MAX_SELECT_COUNT)
        parser.add_argument('--op_internal_size', type=int, default=1028)
