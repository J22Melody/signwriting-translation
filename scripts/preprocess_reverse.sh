#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data_reverse
tools=$base/tools

src=sign
trg=spoken

#################################################################

# BPE by spm
spm_train --input=$data/train.$trg --model_prefix=$data/spm --vocab_size=2000 --model_type bpe --character_coverage 1.0
cat $data/train.$trg | spm_encode --model=$data/spm.model > $data/train.spm.$trg
cat $data/dev.$trg | spm_encode --model=$data/spm.model > $data/dev.spm.$trg
cat $data/test.$trg | spm_encode --model=$data/spm.model > $data/test.spm.$trg

# file sizes
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.$src $data/$corpus.$trg
done

# sanity checks

echo "At this point, please check that 1) file sizes are as expected, 2) languages are correct and 3) material is still parallel"
