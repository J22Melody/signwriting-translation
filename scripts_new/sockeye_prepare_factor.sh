#! /bin/bash

data_dir=$1
data_dir_output=$2
data_dir_pretrained=$3

if [ -n "$data_dir_pretrained" ]; then
  optional_prepare_data_args="--target-vocab $data_dir_pretrained/vocab.trg.0.json \
  --source-vocab $data_dir_pretrained/vocab.src.0.json \
  --source-factor-vocabs $data_dir_pretrained/vocab.src.1.json $data_dir_pretrained/vocab.src.2.json $data_dir_pretrained/vocab.src.3.json $data_dir_pretrained/vocab.src.4.json $data_dir_pretrained/vocab.src.5.json $data_dir_pretrained/vocab.src.6.json $data_dir_pretrained/vocab.src.7.json"
fi

python -m sockeye.prepare_data \
--target $data_dir/train.spm.spoken \
--source $data_dir/train.sign \
--source-factors $data_dir/train.feat_x $data_dir/train.feat_y $data_dir/train.feat_x_rel $data_dir/train.feat_y_rel \
    $data_dir/train.sign+ $data_dir/train.feat_col $data_dir/train.feat_row \
--output $data_dir_output \
--max-seq-len 200 \
--seed 42 \
$optional_prepare_data_args
