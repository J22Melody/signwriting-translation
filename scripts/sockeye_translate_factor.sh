#! /bin/bash

model_name=$1
test_out=models/$model_name/test.hyps

python -m sockeye.translate \
--models models/$model_name \
--input data_reverse/test.spm.spoken \
--output $test_out.mixed \
--max-input-length 99999 \
--beam-size 3 \
--device-id 0 \
--disable-device-locking \
--seed 42 \
--output-type translation_with_factors

python ./scripts/sockeye_postprocess.py $model_name

cat $test_out.symbol | sacrebleu data_reverse/test.symbol -m bleu chrf > $test_out.symbol.eval
