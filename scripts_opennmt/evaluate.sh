#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..
configs=$base/configs_opennmt
models=$base/models_opennmt
translations=$base/translations_opennmt
data=$base/data

mkdir -p $translations

model_name=$1

for checkpoint in $models/$model_name/model_step*.pt; do
    echo "# Translating with checkpoint $checkpoint"
    name=$(basename $checkpoint)
    onmt_translate \
        -gpu 0 \
        -batch_size 32 -batch_type sents \
        -beam_size 5 \
        -model $checkpoint \
        -src $data/test.sign \
        -tgt $data/test.spm.en \
        -output $translations/test.hyp_${name%.*}.en

    echo "detokenize the hypothesis ... "
    spm_decode \
        -model=$data/m.model \
        -input_format=piece \
        < $translations/test.hyp_${name%.*}.en \
        > $translations/test.hyp_${name%.*}.raw.en

    echo "compute detokenized BLEU with sacrebleu ... "
    sacrebleu --lowercase $data/test.en < $translations/test.hyp_${name%.*}.raw.en
done