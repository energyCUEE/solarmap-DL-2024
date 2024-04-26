if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

pred_len=1
label_len=0  
moving_avg=4
batch_size=64
seq_len=37
target=I


# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=MS 
num_features=7 

model_name=Informer 

for d_model in 8 16 32 64 128
do
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/CUEE_PMAS/ \
    --test_data_path pmaps_test_data.csv \
    --train_data_path pmaps_train_data.csv \
    --model_id CUEE_PMAS_$seq_len'_'$pred_len \
    --model $model_name \
    --moving_avg $moving_avg \
    --data CUEE_PMAS \
    --features $feature_type \
    --target $target \
    --seq_len $seq_len \
    --label_len $label_len\
    --pred_len $pred_len \
    --embed_type 4 \
    --e_layers 2 \
    --d_model $d_model \
    --d_target 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $num_features \
    --dec_in $num_features \
    --c_out  $num_features \
    --des 'Exp' \
    --batch_size $batch_size --learning_rate 0.0001 --itr 1  
done
 