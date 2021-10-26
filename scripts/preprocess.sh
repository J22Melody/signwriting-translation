#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data
tools=$base/tools

# mkdir -p $base/model

src=sign
trg=en

# cloned from https://github.com/bricksdont/moses-scripts
MOSES=$tools/moses-scripts/scripts

#################################################################

spm_train --input=$data/train.withDict.en --model_prefix=$data/m --vocab_size=2000 --model_type bpe --character_coverage 1.0 --normalization_rule_name=nmt_nfkc_cf
cat $data/train.withDict.en | spm_encode --model=$data/m.model > $data/train.withDict.spm.en
cat $data/dev.en | spm_encode --model=$data/m.model > $data/dev.spm.en
cat $data/test.en | spm_encode --model=$data/m.model > $data/test.spm.en

# build joeynmt vocab
# python $tools/joeynmt/scripts/build_vocab.py $data/train.$src --output_path $base/model/src_vocab.txt
# python $tools/joeynmt/scripts/build_vocab.py $data/train.$trg --output_path $base/model/trg_vocab.txt

# file sizes
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.$src $data/$corpus.$trg
done

# sanity checks

echo "At this point, please check that 1) file sizes are as expected, 2) languages are correct and 3) material is still parallel"
