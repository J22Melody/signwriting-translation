#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

models=$base/model_$1
configs=$base/configs

mkdir -p $model

num_threads=6
device=5

# CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/baseline.yaml
python -m joeynmt train $configs/$1.yaml
