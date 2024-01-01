#!/bin/bash

# Helper script to convert cosql generated splits to covis.
# Each time the generation script is changed, run this script.

which python
python generate_covis.py --split train
python generate_covis.py --split test
python generate_covis.py --split dev

