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

test_out=$model/best.hyps.test

# # n-best translation
# paste -d'|' $data/test.sign $data/test.sign+ $data/test.feat_col $data/test.feat_row $data/test.feat_x $data/test.feat_y $data/test.feat_x_rel $data/test.feat_y_rel \
# | python -m joeynmt translate $configs/$model_name.yaml -n 5 --ckpt $model/best.ckpt \
# > $test_out

# decode spm
cat $test_out | spm_decode --model=$data/spm.model > $test_out.raw

# split languages
python ./scripts/split_data_by_language.py $model_name

# for sentences: bleu and chrf
for language in 'en-us' 'pt-br' 'mt-mt'; do
    cat $test_out.$language | sacrebleu $data/test.$language -m bleu chrf > $test_out.$language.eval
done

# for dicts: top-5 accuracy
for language in 'dict.en-us' 'dict.en-sg' 'dict.de-de' 'dict.de-ch' 'dict.fr-ca' 'dict.fr-be' 'dict.fr-ch' 'dict.fr-fr' 'dict.pt-br' 'dict.es-es' 'dict.es-hn' 'dict.es-ar' 'dict.es-ni' 'dict.ca-es' 'dict.ar-tn' 'dict.ko-kr' 'dict.mt-mt' 'dict.nl-be' 'dict.pl-pl' 'dict.sk-sk' 'dict.sl-sl'; do
    python ./scripts/top-n-accuracy.py $test_out.$language $data/test.$language > $test_out.$language.eval
done
