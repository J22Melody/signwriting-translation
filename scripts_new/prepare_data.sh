#! /bin/bash

# Usage:
# sh ./scripts_new/prepare_data.sh original data_new_original
# sh ./scripts_new/prepare_data.sh cleaned data_new_cleaned
# sh ./scripts_new/prepare_data.sh expaneded data_new_expaneded

dataset=$1
data_dir=$2

echo "Prepare the $dataset SignBank+ dataset in ./$data_dir ..."

# 1. copy files source and target files to this submodule

mkdir -p $data_dir

cp ../../../data/parallel/$dataset/dev.source ./$data_dir/dev.fsw
cp ../../../data/parallel/$dataset/dev.target ./$data_dir/dev.spoken
cp ../../../data/parallel/$dataset/train.source ./$data_dir/train.fsw
cp ../../../data/parallel/$dataset/train.target ./$data_dir/train.spoken
cp ../../../data/parallel/test/all.source ./$data_dir/test.fsw
cp ../../../data/parallel/test/all.target ./$data_dir/test.spoken

# 2. BPE segmentation on the spoken side

spm_train --input=$data_dir/train.spoken --model_prefix=$data_dir/spm --vocab_size=3000 --model_type bpe
cat $data_dir/train.spoken | spm_encode --model=$data_dir/spm.model > $data_dir/train.spm.spoken
cat $data_dir/dev.spoken | spm_encode --model=$data_dir/spm.model > $data_dir/dev.spm.spoken
cat $data_dir/test.spoken | spm_encode --model=$data_dir/spm.model > $data_dir/test.spm.spoken

# 3. SignWriting factorization on the sign side

python ./scripts_new/factorize_fsw.py --input=./$data_dir/train.fsw
python ./scripts_new/factorize_fsw.py --input=./$data_dir/dev.fsw
python ./scripts_new/factorize_fsw.py --input=./$data_dir/test.fsw
