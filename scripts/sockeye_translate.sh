#! /bin/bash

test_out=models/sockeye_spoken2symbol/test.hyps.symbol

python -m sockeye.translate \
--models models/sockeye_spoken2symbol \
--input data_reverse/test.spm.spoken \
--output $test_out \
--max-input-length 99999 \
--beam-size 3 \
--device-id 0 \
--disable-device-locking \
--seed 42

cat $test_out | sacrebleu data_reverse/test.symbol -m bleu chrf > $test_out.eval
