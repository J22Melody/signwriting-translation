#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..
data=$base/data_reverse
model_name=$1
test_out=models/$model_name/test.hyps

# split languages and parts
python ./scripts/split_data_reverse.py $model_name

# evaluate by language
for language in en pt de fr; do
    cat $test_out.$language.symbol | sacrebleu $data/test.$language.symbol -m bleu chrf --chrf-word-order 2 > $test_out.$language.symbol.eval
done