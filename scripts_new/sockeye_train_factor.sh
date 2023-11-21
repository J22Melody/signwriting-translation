#! /bin/bash

python -m sockeye.train \
--prepared-data data_sockeye_factor_new \
-vt data_new_original/dev.spm.spoken \
-vs data_new_original/dev.sign \
-vsf data_new_original/dev.feat_x data_new_original/dev.feat_y data_new_original/dev.feat_x_rel data_new_original/dev.feat_y_rel \
    data_new_original/dev.sign+ data_new_original/dev.feat_col data_new_original/dev.feat_row \
--output models_new/sockeye_factor \
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
--batch-size 2048 
