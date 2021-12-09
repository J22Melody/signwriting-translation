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

# # decode spm
# cat $test_out | spm_decode --model=$data/spm.model > $test_out.raw

# # split languages
# python ./scripts/split_data_by_language.py $model_name

# # for sentences: bleu and chrf
# for language in en pt; do
#     cat $test_out.$language | sacrebleu $data/test.$language -m bleu chrf > $test_out.$language.eval
# done

# # for dicts: top-5 accuracy
# for language in dict.en dict.de dict.fr dict.pt; do
#     python ./scripts/top-n-accuracy.py $test_out.$language $data/test.$language > $test_out.$language.eval
# done