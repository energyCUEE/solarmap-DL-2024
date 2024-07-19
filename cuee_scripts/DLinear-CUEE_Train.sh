if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=1
label_len=0  
batch_size=32
target=I

seq_len=5
moving_avg=3

model_name=DLinear
# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=MS 
num_features=11  
is_training=1

model_name=DLinear
python -u run_longExp.py \
  --is_training $is_training  \
  --root_path ./dataset/CUEE/ \
  --root_path ./dataset/CUEE_PMAPS_NIGHT/ \
  --test_data_path pmaps_test_with_nighttime.csv \
  --valid_data_path pmaps_validate_with_nighttime.csv \
  --train_data_path pmaps_train_with_nighttime.csv \
  --model_id CUEE_PMAPS_NIGHT_$seq_len'_'$pred_len \
  --model $model_name \
  --moving_avg $moving_avg \
  --data CUEE_PMAPS_NIGHT \
  --features  $feature_type \
  --target    $target \
  --d_target  $d_target\
  --seq_len   $seq_len \
  --label_len $label_len\
  --pred_len  $pred_len \
  --enc_in    $num_features \
  --des 'Exp' \
  --loss 'l1' \
  --scheduler 'ReduceLROnPlateau' \
  --train_epochs 50 \
  --itr 1 --batch_size $batch_size  --learning_rate 0.001 --individual  
  
 