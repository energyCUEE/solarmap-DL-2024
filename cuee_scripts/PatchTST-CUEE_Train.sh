if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=1
label_len=0  
moving_avg=4
batch_size=32
seq_len=4
target=I

# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=MS 
num_features=11 

patch_len=2
stride=1
e_layer=3
model_name=PatchTST  
d_ff=512
d_model=16
e_layer=4

seq_len=(4 8 12 16)
patch_len=(2 4 6 8)
stride=(1 2 3 4)


for i in "${!seq_len[@]}"
do

seq_len_temp=${seq_len[i]}
patch_len_temp=${patch_len[i]}
stride_temp=${stride[i]}

printf '%q %q %q\n' "$seq_len_temp" "$patch_len_temp" "$stride_temp"

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/CUEE_PMAPS_NIGHT/ \
    --test_data_path pmaps_test_with_nighttime.csv \
    --valid_data_path pmaps_validate_with_nighttime.csv \
    --train_data_path pmaps_train_with_nighttime.csv \
    --model_id CUEE_PMAPS_NIGHT_$seq_len_temp'_'$pred_len \
    --model $model_name \
    --moving_avg    $moving_avg \
    --data CUEE_PMAPS_NIGHT \
    --features      $feature_type \
    --target        $target \
    --seq_len       $seq_len_temp \
    --label_len     $label_len\
    --pred_len      $pred_len \
    --enc_in        $num_features \
    --e_layers      $e_layer \
    --n_heads       4 \
    --d_model       $d_model \
    --d_ff          $d_ff \
    --d_target      1\
    --dropout       0.3\
    --fc_dropout    0.3\
    --head_dropout  0\
    --patch_len     $patch_len_temp\
    --stride        $stride_temp \
    --des           'Exp' \
    --train_epochs  10\
    --lradj         'TST'\
    --pct_start     0.3\
    --itr 1 --batch_size $batch_size --learning_rate 0.001
done