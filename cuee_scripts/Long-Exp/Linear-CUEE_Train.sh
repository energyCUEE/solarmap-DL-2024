if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=16
label_len=0
model_name=DLinear
for pred_len in 4
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
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.005 --individual >logs/LongForecasting/$model_name'_I_'CUEE_$seq_len'_'$pred_len.log 
 
done
 