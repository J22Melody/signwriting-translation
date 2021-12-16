#! /bin/bash

test_out=models/sockeye_spoken2symbol/test.hyps.symbol
cat $test_out | sacrebleu data_reverse/test.symbol -m bleu chrf --chrf-word-order 2 > $test_out.eval
