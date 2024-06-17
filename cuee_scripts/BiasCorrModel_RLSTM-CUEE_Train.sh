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
seq_len=5
model_name=BiasCorrModel
feature_type=MS
num_features=10 # len(features_list) -1 --> I_LGBM, Ireg = 11, Inwp = 10, Iclr = 10 
d_model=64 
e_layer=5
moving_avg=4


# option_Ihat1=Iclr # Iclr, Inwp, I_LGBM, Ireg ,if I_LGBM -->  num_features = 11
m2_name=RLSTM
folder_data=true_cloud_relation
checkpoints=checkpoints_true_cloud_relation


for option_Ihat1 in Inwp; do
    python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/${folder_data}/ \
        --test_data_path test_data_true_relation_LGBM_reg.csv \
        --valid_data_path val_data_true_relation_LGBM_reg.csv \
        --train_data_path train_data_true_relation_LGBM_reg.csv \
        --model_id ${folder_data}_${seq_len}'_'${pred_len} \
        --model $model_name \
        --data ${folder_data} \
        --features $feature_type \
        --target $target \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --d_model $d_model \
        --e_layers $e_layer \
        --moving_avg $moving_avg \
        --enc_in $num_features \
        --dec_in $num_features \
        --c_out $num_features \
        --dropout 0.01 \
        --des 'Exp' \
        --loss 'l1' \
        --scheduler 'ReduceLROnPlateau' \
        --train_epochs 100 \
        --batch_size $batch_size \
        --learning_rate 0.001 \
        --itr 1\
        --option_Ihat1 $option_Ihat1\
        --m2_name $m2_name\
        --checkpoints $checkpoints
done
