#! /bin/bash

python -m sockeye.train \
--prepared-data data_sockeye_factor \
-vs data_reverse/dev.spm.spoken \
-vt data_reverse/dev.symbol \
-vtf data_reverse/dev.feat_x data_reverse/dev.feat_y \
--output models/sockeye_spoken2symbol_factor_0.1 \
--overwrite-output \
--weight-tying-type none \
--label-smoothing 0.2 \
--optimized-metric perplexity \
--checkpoint-interval 4000 \
--max-num-checkpoint-not-improved 20 \
--embed-dropout 0.5 \
--transformer-dropout-attention 0.5 \
--initial-learning-rate 0.0001 \
--learning-rate-reduce-factor 0.7 \
--learning-rate-reduce-num-not-improved 5 \
--decode-and-evaluate -1 \
--keep-last-params 1 \
--cache-last-best-params 1 \
--device-id 0 \
--disable-device-locking \
--seed 42 \
--target-factors-num-embed 16 16 \
--target-factors-combine concat \
--target-factors-weight 0.1 \
--batch-size 2048 \
--update-interval 2 \
# --dry-run \
# --use-cpu \

