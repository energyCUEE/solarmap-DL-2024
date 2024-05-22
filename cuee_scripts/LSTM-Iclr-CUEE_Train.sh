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
# seq_len=5
model_name=RLSTM
feature_type=MS
num_features=11
d_model=16
e_layer=1

seq_len=5

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/CUEE_PMAPS_NIGHT/ \
    --test_data_path pmaps_test_with_nighttime.csv \
    --valid_data_path pmaps_validate_with_nighttime.csv \
    --train_data_path pmaps_train_with_nighttime.csv \
    --model_id CUEE_PMAPS_NIGHT_${seq_len}'_'${pred_len} \
    --model $model_name \
    --data CUEE_PMAPS_NIGHT \
    --features $feature_type \
    --target $target \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --d_model $d_model \
    --e_layers $e_layer \
    --enc_in $num_features \
    --dec_in $num_features \
    --c_out $num_features \
    --use_Iclr \
    --input_dropout 0.00 \
    --dropout 0.0 \
    --des 'Exp' \
    --loss 'l1' \
    --scheduler 'ReduceLROnPlateau' \
    --train_epochs 100 \
    --batch_size $batch_size \
    --learning_rate 0.001 \
    --itr 1