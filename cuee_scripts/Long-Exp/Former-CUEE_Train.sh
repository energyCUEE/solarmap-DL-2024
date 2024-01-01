if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
pred_len=4
label_len=0  
moving_avg=37
batch_size=32 
# Autoformer
model_name=Informer
for seq_len in 72 90 108 
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/CUEE/ \
    --data_path updated_measurement_Iclr_new.csv \
    --model_id CUEEData_$seq_len'_'$pred_len \
    --model $model_name \
    --moving_avg $moving_avg \
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
    --c_out  1 \
    --des 'Exp' \
    --batch_size $batch_size --itr 1 >'logs/LongForecasting/'$model_name'_I_CUEE_mv'$moving_avg'_'$seq_len'_'$label_len'_'$pred_len'_'$batch_size'.log' 
done 

model_name=Transformer 
for seq_len in  18 36 54 72 90 108 
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/CUEE/ \
    --data_path updated_measurement_Iclr_new.csv \
    --model_id CUEEData_$seq_len'_'$pred_len \
    --model $model_name \
    --moving_avg $moving_avg \
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
    --c_out  1 \
    --des 'Exp' \
    --batch_size $batch_size --itr 1 >'logs/LongForecasting/'$model_name'_I_CUEE_mv'$moving_avg'_'$seq_len'_'$label_len'_'$pred_len'_'$batch_size'.log' 
done  