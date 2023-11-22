#! /bin/bash

data_dir=$1
data_dir_output=$2

python -m sockeye.prepare_data \
--target $data_dir/train.spm.spoken \
--source $data_dir/train.sign \
--source-factors $data_dir/train.feat_x $data_dir/train.feat_y $data_dir/train.feat_x_rel $data_dir/train.feat_y_rel \
    $data_dir/train.sign+ $data_dir/train.feat_col $data_dir/train.feat_row \
--output $data_dir_output \
--max-seq-len 200 \
--seed 42
