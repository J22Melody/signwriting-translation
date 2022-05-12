#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data_bilingual

# mkdir -p $base/model

src=sign
trg=en

# BPE by spm
# spm_train --input=$data/train.en --model_prefix=$data/spm --vocab_size=2000 --model_type bpe --character_coverage 1.0 --normalization_rule_name nmt_nfkc_cf
spm_train --input=$data/train.en --model_prefix=$data/spm --vocab_size=2000 --model_type bpe --character_coverage 1.0
cat $data/train.en | spm_encode --model=$data/spm.model > $data/train.spm.en
cat $data/dev.en | spm_encode --model=$data/spm.model > $data/dev.spm.en
cat $data/test.en | spm_encode --model=$data/spm.model > $data/test.spm.en

# file sizes
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.$src $data/$corpus.$trg
done

# sanity checks

echo "At this point, please check that 1) file sizes are as expected, 2) languages are correct and 3) material is still parallel"