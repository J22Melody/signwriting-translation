#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data_bilingual
configs=$base/configs

src=sign
trg=en

model_name=$1
model=$base/models/$model_name

python -m joeynmt test $configs/$model_name.yaml --ckpt $model/best.ckpt --output_path $model/best.hyps

for test_out in $model/*.hyps.test; do	
cat $test_out | sacrebleu --lowercase --chrf-lowercase $data/test.spm.$trg -m bleu chrf > $test_out.eval
cat $test_out | spm_decode --model=$data/spm.model > $test_out.raw
cat $test_out.raw | sacrebleu --lowercase --chrf-lowercase $data/test.$trg -m bleu chrf > $test_out.raw.eval
done