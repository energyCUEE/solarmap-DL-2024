if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=4 
label_len=0

for model_name in  Linear NLinear DLinear 
do
for seq_len in 24 128
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
  --itr 1 --batch_size 128  --learning_rate 0.005 --individual >logs/LongForecasting/$model_name'_I_'CUEE_$seq_len'_'$label_len'_'$pred_len.log 
 
done
done
 