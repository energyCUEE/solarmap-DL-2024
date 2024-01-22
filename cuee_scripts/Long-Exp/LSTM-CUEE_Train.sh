if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

d_model=64
enc_in=1
pred_len=4 
label_len=0
moving_avg=37 
batch_size=128
model_name=RLSTM

for seq_len in 18 36 54 72 90 108 
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/CUEE/ \
  --data_path updated_measurement_Iclr_new.csv \
  --model_id CUEEData_$seq_len'_'$pred_len \
  --model $model_name \
  --data CUEE \
  --features S \
  --target I \
  --seq_len $seq_len \
  --label_len $label_len\
  --pred_len $pred_len \
  --enc_in $enc_in \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 --batch_size $batch_size  --learning_rate 0.0001 > 'logs/LongForecasting/'$model_name'_I_CUEE_'$seq_len'_'$label_len'_'$pred_len'_en'$enc_in'_hd'$d_model'_'$batch_size'.log' 

done