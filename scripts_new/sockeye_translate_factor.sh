#! /bin/bash

model_name=$1
test_out=models_new/$model_name/test.hyps

python -m sockeye.translate \
--models models_new/$model_name \
--input data_new_original/test.sign \
--input-factors data_new_original/test.feat_x data_new_original/test.feat_y data_new_original/test.feat_x_rel data_new_original/test.feat_y_rel \
    data_new_original/test.sign+ data_new_original/test.feat_col data_new_original/test.feat_row \
--output $test_out.spm.spoken \
--max-input-length 99999 \
--beam-size 5 \
--device-id 0 \
--brevity-penalty-type constant \
--seed 42 

cat $test_out.spm.spoken | sacrebleu data_new_original/test.spm.spoken -m bleu chrf > $test_out.spm.spoken.eval
cat $test_out.spm.spoken | spm_decode --model=data_new_original/spm.model > $test_out.spoken
cat $test_out.spoken | sacrebleu data_new_original/test.spoken -m bleu chrf > $test_out.spoken.eval
