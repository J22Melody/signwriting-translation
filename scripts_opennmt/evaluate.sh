#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..
configs=$base/configs_opennmt
models=$base/models_opennmt
translations=$base/translations_opennmt
data=$base/data

model_name=$1

mkdir -p $translations
mkdir -p $translations/$model_name

for checkpoint in $models/$model_name/model_step*.pt; do
    # echo "# Translating with checkpoint $checkpoint"
    # name=$(basename $checkpoint)
    # onmt_translate \
    #     -gpu 0 \
    #     -batch_size 32 -batch_type sents \
    #     -beam_size 5 \
    #     -length_penalty wu \
    #     -alpha 1 \
    #     -model $checkpoint \
    #     -src $data/test.sign \
    #     -tgt $data/test.spm.en \
    #     -output $translations/$model_name/test.hyp_${name%.*}.en
    #     --src_feats "{'feat_col': '$data/test.sign+.feat_col', 'feat_row': '$data/test.sign+.feat_row', 'feat_x': '$data/test.sign+.feat_x', 'feat_y': '$data/test.sign+.feat_y',"

    # echo "compute BLEU with sacrebleu ... "
    # sacrebleu --lowercase $data/test.spm.en < $translations/$model_name/test.hyp_${name%.*}.en

    echo "detokenize the hypothesis ... "
    spm_decode \
        -model=$data/m.model \
        -input_format=piece \
        < $translations/test.hyp_${name%.*}.en \
        > $translations/test.hyp_${name%.*}.raw.en

    echo "compute detokenized BLEU with sacrebleu ... "
    sacrebleu --lowercase $data/test.en < $translations/test.hyp_${name%.*}.raw.en
done