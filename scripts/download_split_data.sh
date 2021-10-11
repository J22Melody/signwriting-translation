#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data

mkdir -p $data
mkdir -p $data/dev

wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz -P $data
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz -P $data
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz -P $data

cat $data/corpus.tc.de.gz | gunzip -c - > $data/train.truecased.de
cat $data/corpus.tc.en.gz | gunzip -c - > $data/train.truecased.en

tar -xzvf $data/dev.tgz -C $data/dev

cp $data/dev/newstest2015.tc.de $data/dev.truecased.de
cp $data/dev/newstest2015.tc.en $data/dev.truecased.en

cp $data/dev/newstest2016.tc.de $data/test.truecased.de
cp $data/dev/newstest2016.tc.en $data/test.truecased.en

# sizes
echo "Sizes of corpora:"
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.truecased.de $data/$corpus.truecased.en
done

# sanity checks
echo "At this point, please make sure that 1) number of lines are as expected, 2) language suffixes are correct and 3) files are parallel"
