#!/bin/bash

# This shell script is a utility to generate the pickle files for train and test sets.

cd ../generating_queries

if [ ! -d "train_pickle" ]

then

  mkdir train_pickle

fi

python3 generate_training_tuples_RGB_baseline_batch.py
