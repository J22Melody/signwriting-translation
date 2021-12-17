#! /bin/bash

python -m sockeye.prepare_data \
--source data_reverse/train.spm.spoken \
--target data_reverse/train.symbol \
--target-factors data_reverse/train.feat_x data_reverse/train.feat_y \
--output ./data_sockeye_factor \
--max-seq-len 200 \
--seed 42
