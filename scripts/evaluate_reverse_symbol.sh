#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data_reverse
configs=$base/configs

src=spm.spoken
trg=sign

model_name=$1
model=$base/models/$model_name

test_out=$model/best.hyps.test

# translation
python -m joeynmt translate $configs/$model_name.yaml --ckpt $model/best.ckpt \
< $data/test.$src > $test_out
# python -m joeynmt test $configs/$model_name.yaml --ckpt $model/best.ckpt

# evaluate altogether
# cat $test_out | sacrebleu $data/test.$trg -m bleu chrf --chrf-word-order 2 > $test_out.eval

# # split languages and parts
# python ./scripts/split_data_reverse.py $model_name

# # evaluate symbols
# for language in en pt dict.en dict.de dict.fr dict.pt; do
#     cat $test_out.$language.sym.$trg | sacrebleu $data/test.$language.sym.$trg -m bleu chrf --chrf-word-order 2 > $test_out.$language.sym.eval
# done

# evaluate numbers

# # convert back to raw
# for language in dict.en dict.de dict.fr dict.pt; do
#     python ./scripts/top-n-accuracy.py $test_out.$language $data/test.$language > $test_out.$language.eval
# done