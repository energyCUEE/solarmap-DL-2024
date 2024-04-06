if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

 
pred_len=15 
label_len=0

moving_avg=37 
batch_size=128
seq_len=74
model_name=RLSTM
target=I

# we run only two mode "S" or "MS"; if you use "S", please change to num_feature=1 
feature_type=S 
num_features=1 
mode=test # test val 

# 8 16 32 64 128 256 512
for d_model in 16 128 256
do 
    python -u plot_longExp.py \
        --mode $mode \
        --root_path ./dataset/CUEERoof/ \
        --data_path "EE-Station-20230101-20231231_rmNan.csv" \
        --model_id 'CUEEData_'$seq_len'_'$pred_len \
        --model $model_name \
        --moving_avg $moving_avg \
        --data CUEE \
        --features $feature_type \
        --target $target \
        --seq_len $seq_len \
        --label_len $label_len\
        --pred_len $pred_len \
        --enc_in $num_features \
        --d_model $d_model \
        --des 'Exp' \
        --itr 1 --batch_size $batch_size  --learning_rate 0.0001 # >'logs/LongForecasting/'$model_name"-d"$d_model"_"$feature_type"_"$target"_mv"$moving_avg"_CUEE_"$seq_len'_'$label_len'_'$pred_len'_'$batch_size'_infer.log' 
done  