from torch.utils.data import DataLoader
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.models import builder as model_builder
from src.datasets import CoVis, collate, transforms
from src.config import setup_training_argparse, trainer_args

# Reproducibility
pl.seed_everything(5, workers=True)

"""
Main training script. See setup_training_argparse
Each model also has its own args as well. PyTorch lightning has some args to configure as well.
"""

# args
parser = argparse.ArgumentParser('Train Script')
parser = setup_training_argparse(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# load model
model, task = model_builder.get_model(args)

# Setup logging
logger = TensorBoardLogger('tb_logs', name=task)

# Load datasets and dataloaders
# temporarily change to test for quick debug
train_set = CoVis(CoVis.train_split,
                  utterance_transforms=transforms.
                  default_utterance_transforms(),
                  where_transforms=transforms.default_where_transform(),
                  column_transforms=transforms.default_column_transforms(),
                  context_sentences=args.context_sentences,
                  shuffle_columns=False, oversampling_target = task)

dev_set = CoVis(CoVis.dev_split,
                utterance_transforms=transforms.default_utterance_transforms(),
                where_transforms=transforms.default_where_transform(),
                column_transforms=transforms.default_column_transforms(),
                context_sentences=args.context_sentences, oversampling_target = task)

print("dev size:", len(dev_set))

train_dloader = DataLoader(
    train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=args.num_dataloading_workers,
    collate_fn= collate
)

dev_dloader = DataLoader(
    dev_set,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=args.num_dataloading_workers,
    collate_fn=collate
)

# Model checkpointing

checkpoint_callback = ModelCheckpoint(
    filename='best_model', monitor='loss/val')

if args.save_using_acc:
    checkpoint_callback = ModelCheckpoint(
        filename='best_model', monitor='accuracy/val', mode='max')

trainer: pl.Trainer = pl.Trainer.from_argparse_args(
    args, callbacks=[checkpoint_callback], logger=logger, **trainer_args)
trainer.fit(model, train_dloader, dev_dloader)
