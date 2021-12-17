#! /bin/bash

python -m sockeye.prepare_data \
--source data_reverse/train.spm.spoken \
--target data_reverse/train.symbol \
--output ./data_sockeye \
--max-seq-len 200 \
--seed 42
