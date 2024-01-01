from abc import abstractmethod, abstractstaticmethod
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics import Metric


class BaseModel(pl.LightningModule):
    '''
    Generic model class which all other classes inherit from.
    Provides helpful utilities and abstract methods.
    Also runs the pytorch lightning necessary setup.
    '''

    def __init__(self, learning_rate,
                 bert_learning_rate,
                 loss_fn=nn.CrossEntropyLoss()):
        super(BaseModel, self).__init__()
        self.accuracy_metric = AccuracyMetric()
        self.loss_metric = LossMetric()
        self.loss_fn = loss_fn
        self.heads = []  # Heads to feed bert predictions in
        self.bert_model = None
        # Bert and the heads have different learning rates
        self.learning_rate = learning_rate
        self.bert_learning_rate = bert_learning_rate
        self.val_loss = LossMetric()
        self.val_acc = AccuracyMetric()
        self.train_loss = LossMetric()
        self.train_acc = AccuracyMetric()

    @property
    def device(self):
        return next(self.parameters()).device

    def get_heads(self):
        return self.heads

    def get_head_params(self):
        final_params = []
        for head in self.heads:
            final_params += list(head.parameters())
        return final_params

    def bert_forward(self, utterances, columns):
        utterance_inputs = utterances['input_ids']
        utterance_mask = utterances['attention_mask']
        col_inputs = columns['input_ids']
        col_mask = columns['attention_mask']

        input_ids = torch.hstack(
            [utterance_inputs, col_inputs])
        attention_mask = torch.hstack(
            [utterance_mask, col_mask])

        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        cls_tokens = bert_output['pooler_output']
        bert_output = bert_output['last_hidden_state']
        return cls_tokens, bert_output

    def get_bert_params(self):
        if type(self.bert_model) is list:
            params = []
            for m in self.bert_model:
                params += list(m.parameters())
            return params
        else:
            return self.bert_model.parameters()

    def _extract_column_headers(self, activations, col_pointers):
        '''Helper method to extract activations corresponding to pointers.'''
        out = []
        for i in range(activations.shape[0]):
            out.append(activations[i][col_pointers[i]])
        out = torch.stack(out)
        return out

    def configure_optimizers(self):
        '''Default optimizers'''
        return Adam([
            {'params': self.get_bert_params(), 'lr': self.bert_learning_rate},
            {'params': self.get_head_params(), 'lr': self.learning_rate}
        ])

    def _common_step(self, batch, stage):
        '''Step for inference, train, and test'''
        out = self(batch)
        loss = self.loss(out, batch)
        correctness_vector = self.accuracy_internal(out, batch)
        # Log the metrics for both loss and accuracy
        if stage == "val":
            self.val_acc(correctness_vector)
            self.val_loss(loss, correctness_vector.numel())
        elif stage == 'train':
            self.train_acc(correctness_vector)
            self.train_loss(loss, correctness_vector.numel())

        self.log(f'accuracy/{stage}', self.val_acc if stage == 'val' else self.train_acc, on_step=False,
                 on_epoch=True, prog_bar=True,
                 batch_size=self._get_batch_size(batch))

        self.log(f'loss/{stage}', self.val_loss if stage == 'val' else self.train_loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 batch_size=self._get_batch_size(batch))
        return out, loss

    def _on_epoch_end_generic(self, stage):
        pass

    def training_epoch_end(self, outputs) -> None:
        self._on_epoch_end_generic('train')

    def validation_epoch_end(self, outputs) -> None:
        self._on_epoch_end_generic('val')

    def _get_batch_size(self, batch):
        # Given a batch, returns an int for the batch size
        return batch[0]['input_ids'].shape[0]

    @abstractmethod
    def loss(self, model_out, batch):
        '''Loss definition'''
        pass

    def accuracy(self, model_out, batch):
        '''Wrapper of accuracy internal'''
        correctness_vector = self.accuracy_internal(model_out, batch)
        return correctness_vector.sum() / correctness_vector.shape[0]

    @abstractmethod
    def accuracy_internal(self, model_out, batch):
        '''Accuracy definition'''
        pass

    def training_step(self, batch, batch_idx):
        out, loss = self._common_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._common_step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        out, loss = self._common_step(batch, 'test')
        return loss

    def predict_step(self, batch, batch_idx):
        '''Predict step is purely used for computing accuracy metrics'''
        out = self(batch)
        raw_accuracy, _ = self.accuracy_internal(out, batch)
        return {'raw_accuracy': raw_accuracy, 'debug_info': batch[-1]}

    def one_hot_encoding(self, targets, max_length):
        new_target = torch.zeros(max_length)
        for i in targets:
            new_target[i] = 1
        return new_target

    @abstractstaticmethod
    def add_argparse_args(subparser):
        '''Definition of additional model args needed.'''
        pass


class AccuracyMetric(Metric):
    '''Helper metrics class for accuracy.'''
    full_state_update: bool = True

    def __init__(self, dist_sync_on_step=False):
        super(AccuracyMetric, self).__init__(
            dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, correctness: torch.Tensor):
        self.correct += correctness.sum().long()
        self.total += correctness.numel()

    def compute(self):
        return self.correct.float() / self.total


class LossMetric(Metric):
    '''Helper metrics class for loss.'''
    full_state_update: bool = True

    def __init__(self, dist_sync_on_step=False):
        super(LossMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("measure", default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: float, total: int):
        self.measure += loss
        self.total += total

    def compute(self):
        return self.measure.float() / self.total
