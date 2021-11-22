#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data
configs=$base/configs

translations=$base/translations

mkdir -p $translations

src=sign
trg=spoken

# cloned from https://github.com/bricksdont/moses-scripts
MOSES=$base/tools/moses-scripts/scripts

num_threads=6
device=5

model_name=$1
model=$base/models/$model_name

paste -d'|' $data/test.sign $data/test.sign+ $data/test.feat_col $data/test.feat_row $data/test.feat_x $data/test.feat_y $data/test.feat_x_rel $data/test.feat_y_rel \
| python -m joeynmt translate $configs/$model_name.yaml -n 5 --ckpt $model/best.ckpt \
> $model/best1.hyps.test

# test_out=$model/best.hyps.test

# cat $test_out | sacrebleu $data/test.spm.$trg -m bleu chrf > $test_out.eval
# cat $test_out | spm_decode --model=$data/spm.model > $test_out.raw
# cat $test_out.raw | sacrebleu $data/test.$trg -m bleu chrf > $test_out.raw.eval

# python ./scripts/split_data_by_language.py

# for language in dict.fr en pt dict.de dict.en dict.pt; do
#     cat $test_out.raw.$language | sacrebleu $data/test.$language -m bleu chrf > $test_out.raw.$language.eval
# done