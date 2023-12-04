if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
pred_len=4
label_len=0 


for seq_len in 24 128 
do
for model_name in  Autoformer Informer Transformer 
do 
  python -u plot_longExp.py \
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
    --embed_type 4 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1 >logs/LongForecasting/$model_name'_I_'CUEE_$seq_len'_'$label_len'_'$pred_len.log 
done 
model_name=DLinear
python -u plot_longExp.py \
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
  --itr 1   --learning_rate 0.005 --individual >logs/LongForecasting/$model_name'_I_'CUEE_$seq_len'_'$label_len'_'$pred_len.log 
done 