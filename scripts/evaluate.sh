#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data
configs=$base/configs

translations=$base/translations

mkdir -p $translations

src=sign
trg=en

# cloned from https://github.com/bricksdont/moses-scripts
MOSES=$base/tools/moses-scripts/scripts

num_threads=6
device=5

model_name=$1
model=$base/models/$model_name

python -m joeynmt test $configs/$model_name.yaml --ckpt $model/best.ckpt --output_path $model/best.hyps

# for test_out in $model/*.hyps.test; do	
# cat $test_out | sacrebleu $data/test.spm.$trg -m bleu chrf > $test_out.eval
# cat $test_out | spm_decode --model=$data/spm.model > $test_out.raw
# cat $test_out.raw | sacrebleu $data/test.$trg -m bleu chrf > $test_out.raw.eval
# done