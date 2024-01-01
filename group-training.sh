#!/bin/bash

# Quick and dirty training script. Runs with default hparams.

conda env list

which python

python train.py select_count 
python train.py select_col
python train.py where_count 
python train.py where_operator
python train.py where_col

echo "Completed training."
nvidia-smi
