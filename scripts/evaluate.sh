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

# CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt translate $configs/$model_name.yaml < $data/test.$src > $translations/test.$model_name.$trg
# python -m joeynmt translate $configs/$model_name.yaml --ckpt $model/best.ckpt > $translations/test.$model_name.$trg
# python -m joeynmt test $configs/$model_name.yaml --ckpt $model/best.ckpt

# TODO: recasing
# For now, compute case-insensitive BLEU by passing --lowercase to sacreBLEU

# cat $translations/test.$model_name.$trg | $MOSES/recaser/truecase.perl --model $MOSES/recaser/truecase.model > $translations/test.$model_name.truecased.$trg

# undo tokenization

# cat $translations/test.$model_name.$trg | $MOSES/tokenizer/detokenizer.perl -l $trg > $translations/test.$model_name.raw.$trg
# cat $translations/test.$model_name.$trg | spm_decode --model=$data/m.model > translations/test.$model_name.raw.$trg

# compute case-insensitive BLEU on detokenized data

# cat $translations/test.$model_name.$trg | sacrebleu --lowercase $data/test.tokenized.$trg > $translations/test.$model_name.bleu
# cat $translations/test.$model_name.raw.$trg | sacrebleu --lowercase $data/test.$trg > $translations/test.$model_name.raw.bleu
# cat $translations/test.$model_name.$trg | sacrebleu --lowercase $data/test.spm.$trg > $translations/test.$model_name.bleu
# cat $translations/test.$model_name.raw.$trg | sacrebleu --lowercase $data/test.$trg > $translations/test.$model_name.raw.bleu

# python -m joeynmt test $configs/$model_name.yaml --ckpt $model/best.ckpt --output_path $model/best.hyps

for test_out in $model/*.hyps.test; do	
cat $test_out | spm_decode --model=$data/m.model > $test_out.raw
cat $test_out | sacrebleu --lowercase --chrf-lowercase $data/test.spm.$trg -m bleu chrf > $test_out.eval
cat $test_out.raw | sacrebleu --lowercase --chrf-lowercase $data/test.$trg -m bleu chrf > $test_out.raw.eval
done