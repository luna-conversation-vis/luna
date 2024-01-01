#!/bin/bash

# Converts cosql splits [train, dev] to CoVis splits [train, dev, trian]
# This script only needs to be run once unless the method of generating splits is changed.

mkdir -p data/covis

echo Archiving standard versions of cosql
mv data/cosql_dataset/sql_state_tracking/cosql_dev.json data/cosql_dataset/sql_state_tracking/_cosql_dev.json

echo Running python script to generate data splits
python split_cosql.py
