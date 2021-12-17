#! /bin/bash

test_out=models/sockeye_spoken2symbol_factor/test.hyps

python -m sockeye.translate \
--models models/sockeye_spoken2symbol_factor \
--input data_reverse/test.spm.spoken \
--output $test_out.mixed \
--max-input-length 99999 \
--beam-size 3 \
--device-id 0 \
--disable-device-locking \
--seed 42 \
--output-type translation_with_factors

python ./scripts/sockeye_postprocess.py

cat $test_out.symbol | sacrebleu data_reverse/test.symbol -m bleu chrf > $test_out.symbol.eval
