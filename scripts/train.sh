#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

num_threads=6
device=5

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/transformer_wmt17_ende.yaml
