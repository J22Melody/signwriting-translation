#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

# data=$base/data
data=$base/data

mkdir -p $data

python ./scripts/fetch_data.py

# sizes
echo "Sizes of corpora:"
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.sign $data/$corpus.en
done

# sanity checks
echo "At this point, please make sure that 1) number of lines are as expected, 2) language suffixes are correct and 3) files are parallel"
