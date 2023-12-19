#! /bin/bash

dataset=$1
data_dir=$2
data_dir_pretrained=$3

echo "Prepare the $dataset SignBank+ dataset in ./$data_dir ..."

# 1. copy files source and target files to this submodule

mkdir -p $data_dir

cp ../../../data/parallel/$dataset/dev.source ./$data_dir/dev.fsw
cp ../../../data/parallel/$dataset/dev.target ./$data_dir/dev.spoken
cp ../../../data/parallel/$dataset/train.source ./$data_dir/train.fsw
cp ../../../data/parallel/$dataset/train.target ./$data_dir/train.spoken
cp ../../../data/parallel/test/test.source.unique ./$data_dir/test.fsw

# 2. BPE segmentation on the spoken side

if [ -n "$data_dir_pretrained" ]; then
    cp $data_dir_pretrained/spm.model $data_dir/spm.model
else
    spm_train --input=$data_dir/train.spoken --model_prefix=$data_dir/spm --vocab_size=3000 --model_type bpe
fi
cat $data_dir/train.spoken | spm_encode --model=$data_dir/spm.model > $data_dir/train.spm.spoken
cat $data_dir/dev.spoken | spm_encode --model=$data_dir/spm.model > $data_dir/dev.spm.spoken

# 3. SignWriting factorization on the sign side

python ./scripts_new/factorize_fsw.py --input=./$data_dir/train.fsw
python ./scripts_new/factorize_fsw.py --input=./$data_dir/dev.fsw
python ./scripts_new/factorize_fsw.py --input=./$data_dir/test.fsw
