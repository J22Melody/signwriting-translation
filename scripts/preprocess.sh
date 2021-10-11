#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data
tools=$base/tools

mkdir -p $base/model

src=sign
trg=en

# cloned from https://github.com/bricksdont/moses-scripts
MOSES=$tools/moses-scripts/scripts

#################################################################

# input files are preprocessed already

# remove preprocessing for target language test data, for evaluation

cat $data/test.en | $MOSES/tokenizer/detokenizer.perl -l en > $data/test.raw.en

# build joeynmt vocab
python $tools/joeynmt/scripts/build_vocab.py $data/train.$src --output_path $base/model/src_vocab.txt
python $tools/joeynmt/scripts/build_vocab.py $data/train.$trg --output_path $base/model/trg_vocab.txt

# file sizes
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.$src $data/$corpus.$trg
done

# sanity checks

echo "At this point, please check that 1) file sizes are as expected, 2) languages are correct and 3) material is still parallel"
