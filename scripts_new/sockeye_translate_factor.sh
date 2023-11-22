#! /bin/bash

data_dir=$1
model_name=$2
test_source_dir=${3:-'../../../data/parallel/test'}
test_out=models_new/$model_name/test.hyps

python -m sockeye.translate \
--models models_new/$model_name \
--input $data_dir/test.sign \
--input-factors $data_dir/test.feat_x $data_dir/test.feat_y $data_dir/test.feat_x_rel $data_dir/test.feat_y_rel \
    $data_dir/test.sign+ $data_dir/test.feat_col $data_dir/test.feat_row \
--output $test_out.spm.spoken \
--max-input-length 99999 \
--beam-size 5 \
--device-id 0 \
--brevity-penalty-type constant \
--seed 42 

# cat $test_out.spm.spoken | sacrebleu $data_dir/test.spm.spoken -m bleu chrf > $test_out.spm.spoken.eval
cat $test_out.spm.spoken | spm_decode --model=$data_dir/spm.model > $test_out.spoken
sacrebleu $(find $test_source_dir -type f -name "test.target*") -i $test_out.spoken -m bleu chrf --width 2 > $test_out.spoken.eval
