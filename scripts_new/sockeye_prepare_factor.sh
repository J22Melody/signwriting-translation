#! /bin/bash

python -m sockeye.prepare_data \
--target data_new_original/train.spm.spoken \
--source data_new_original/train.sign \
--source-factors data_new_original/train.feat_x data_new_original/train.feat_y data_new_original/train.feat_x_rel data_new_original/train.feat_y_rel \
    data_new_original/train.sign+ data_new_original/train.feat_col data_new_original/train.feat_row \
--output ./data_sockeye_factor_new \
--max-seq-len 200 \
--seed 42
