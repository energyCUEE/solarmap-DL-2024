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
seq_len=4
target=I


# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=MS 
num_features=11 

model_name=RLSTM 
mode=test # test val 

d_model=50

e_layer=1

for e_layer in 1 2 10 25 50
do
python -u infer_longExp.py \
    --mode $mode \
    --root_path ./dataset/CUEE_PMAPS/ \
    --test_data_path pmaps_test_data.csv \
    --valid_data_path pmaps_validate_data.csv \
    --train_data_path pmaps_train_data.csv \
    --model_id CUEE_PMAPS_$seq_len'_'$pred_len \
    --model $model_name \
    --moving_avg $moving_avg \
    --data CUEE_PMAPS \
    --features $feature_type \
    --target $target \
    --seq_len $seq_len \
    --label_len $label_len\
    --pred_len $pred_len \
    --d_model $d_model \
    --e_layers $e_layer \
    --enc_in $num_features \
    --dec_in $num_features \
    --c_out  $num_features \
    --des 'Exp' \
    --batch_size $batch_size --learning_rate 0.0001 --itr 1   
done