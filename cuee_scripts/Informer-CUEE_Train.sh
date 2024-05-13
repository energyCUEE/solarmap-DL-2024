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

model_name=Informer 
e_layer=4
embed_type=4
is_training=1 # Set to 0 to train... 

for d_model in 16
do
python -u run_longExp.py \
    --is_training $is_training \
    --root_path ./dataset/CUEE_PMAPS_NIGHT/ \
    --test_data_path pmaps_test_with_nighttime.csv \
    --valid_data_path pmaps_validate_with_nighttime.csv \
    --train_data_path pmaps_train_with_nighttime.csv \
    --model_id CUEE_PMAPS_NIGHT_$seq_len'_'$pred_len \
    --model $model_name \
    --moving_avg $moving_avg \
    --data CUEE_PMAPS_NIGHT \
    --features   $feature_type \
    --target     $target \
    --seq_len    $seq_len \
    --label_len  $label_len\
    --pred_len   $pred_len \
    --embed_type $embed_type \
    --e_layers   $e_layer \
    --d_model    $d_model \
    --d_target   1 \
    --d_layers   1 \
    --factor     3 \
    --enc_in $num_features \
    --dec_in $num_features \
    --c_out  $num_features \
    --des 'Exp' \
    --loss 'l1' \
    --scheduler 'ReduceLROnPlateau' \
    --train_epochs 10 \
    --batch_size $batch_size --learning_rate 0.001 --itr 1  
done
 