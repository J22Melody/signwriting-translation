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
python -m joeynmt translate $configs/$model_name.yaml --ckpt $model/best.ckpt < $data/test.$src > $translations/test.$model_name.$trg

# TODO: recasing
# For now, compute case-insensitive BLEU by passing --lowercase to sacreBLEU

# cat $translations/test.$model_name.$trg | $MOSES/recaser/truecase.perl --model $MOSES/recaser/truecase.model > $translations/test.$model_name.truecased.$trg

# undo tokenization

cat $translations/test.$model_name.$trg | $MOSES/tokenizer/detokenizer.perl -l $trg > $translations/test.$model_name.raw.$trg

# compute case-sensitive BLEU on detokenized data

cat $translations/test.$model_name.$trg | sacrebleu --lowercase $data/test.$trg
cat $translations/test.$model_name.raw.$trg | sacrebleu --lowercase $data/test.raw.$trg
		
