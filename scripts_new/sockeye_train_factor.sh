#! /bin/bash

data_dir=$1
data_dir_prepared=$2
model_name=$3
model_pretrained_name=$4

if [ -n "$model_pretrained_name" ]; then
  optional_training_args="--params models_new/$model_pretrained_name/params.best"
fi

python -m sockeye.train \
--prepared-data $data_dir_prepared \
-vt $data_dir/dev.spm.spoken \
-vs $data_dir/dev.sign \
-vsf $data_dir/dev.feat_x $data_dir/dev.feat_y $data_dir/dev.feat_x_rel $data_dir/dev.feat_y_rel \
    $data_dir/dev.sign+ $data_dir/dev.feat_col $data_dir/dev.feat_row \
--output models_new/$model_name \
--overwrite-output \
--weight-tying-type trg_softmax \
--label-smoothing 0.2 \
--optimized-metric bleu \
--checkpoint-interval 4000 \
--update-interval 2 \
--max-num-epochs 300 \
--max-num-checkpoint-not-improved 10 \
--embed-dropout 0.5 \
--transformer-dropout-attention 0.5 \
--initial-learning-rate 0.0001 \
--learning-rate-reduce-factor 0.7 \
--learning-rate-reduce-num-not-improved 5 \
--decode-and-evaluate 500 \
--keep-last-params 1 \
--cache-last-best-params 1 \
--device-id 0 \
--seed 42 \
--source-factors-num-embed 16 16 16 16 16 16 16 \
--source-factors-combine concat \
--batch-size 2048 \
$optional_training_args
