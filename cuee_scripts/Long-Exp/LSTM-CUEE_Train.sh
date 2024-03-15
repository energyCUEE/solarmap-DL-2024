if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


enc_in=1
pred_len=4 
label_len=0

d_model=64

moving_avg=37 
batch_size=128
model_name=RLSTM
target=I

# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=MS 
num_features=9 

for seq_len in 18 36 54 72 90 108 
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/CUEE/ \
  --data_path updated_measurement_Iclr_new.csv \
  --model_id CUEEData_$seq_len'_'$pred_len \
  --model $model_name \
  --data CUEE \
  --features $feature_type \
  --target $target \
  --seq_len $seq_len \
  --label_len $label_len\
  --pred_len $pred_len \
  --enc_in $num_features \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 --batch_size $batch_size  --learning_rate 0.0001 >'logs/LongForecasting/'$model_name"_"$feature_type"_"$target"_mv"$moving_avg"_CUEE_"$seq_len'_'$label_len'_'$pred_len'_'$batch_size'.log' 

done