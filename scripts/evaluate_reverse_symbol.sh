#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data_reverse
configs=$base/configs

src=spm.spoken
trg=symbol

model_name=$1
model=$base/models/$model_name

test_out=$model/best.hyps.test

# translation
# python -m joeynmt translate $configs/$model_name.yaml --ckpt $model/best.ckpt \
# < $data/test.$src > $test_out
# python -m joeynmt test $configs/$model_name.yaml --ckpt $model/best.ckpt --output_path $model/best.hyps

# evaluate altogether
cat $test_out | sacrebleu $data/test.$trg -m bleu chrf --chrf-word-order 2 > $test_out.eval
