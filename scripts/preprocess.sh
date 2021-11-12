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

# truecasing by moses-scripts
# $MOSES/recaser/train-truecaser.perl --model $data/truecase_model --corpus $data/train.en
# $MOSES/recaser/truecase.perl --model $data/truecase_model < $data/train.en > $data/train.truecased.en
# $MOSES/recaser/truecase.perl --model $data/truecase_model < $data/dev.en > $data/dev.truecased.en
# $MOSES/recaser/truecase.perl --model $data/truecase_model < $data/test.en > $data/test.truecased.en

# train recaser by moses-scripts
# $MOSES/recaser/train-recaser.perl --dir $data/recaser_model --corpus $data/train.tokenized.en --first-step 2

# BPE by spm
# spm_train --input=$data/train.en --model_prefix=$data/spm --vocab_size=2000 --model_type bpe --character_coverage 1.0 --normalization_rule_name nmt_nfkc_cf
spm_train --input=$data/train.en --model_prefix=$data/spm --vocab_size=2000 --model_type bpe --character_coverage 1.0
cat $data/train.en | spm_encode --model=$data/spm.model > $data/train.spm.en
cat $data/dev.en | spm_encode --model=$data/spm.model > $data/dev.spm.en
cat $data/test.en | spm_encode --model=$data/spm.model > $data/test.spm.en

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
