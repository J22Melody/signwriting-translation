#! /bin/bash

model_name=$1
test_out=models/$model_name/test.hyps

cat $test_out.symbol | sacrebleu data_reverse/test.symbol -m bleu chrf --chrf-word-order 2 > $test_out.symbol.eval
python ./scripts/sockeye_evaluate_xy.py $test_out.feat_x data_reverse/test.feat_x > $test_out.feat_x.eval
python ./scripts/sockeye_evaluate_xy.py $test_out.feat_y data_reverse/test.feat_y > $test_out.feat_y.eval