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
model_name=Transformer
feature_type=MS
num_features=8 
d_model=16 
e_layer=2
embed_type=0
moving_avg=4
num_features_overlap=7 


folder_data=true_cloud_relation_08JUL24
checkpoints=checkpoints_true_cloud_relation_08JUL24
m2_name=Transformer

for option_Ihat1 in I_R_wo_nwp; do
    python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/$folder_data/ \
        --test_data_path test_data_true_relation_lgbm_new_arrange.csv \
        --valid_data_path val_data_true_relation_lgbm_new_arrange.csv \
        --train_data_path train_data_true_relation_lgbm_new_arrange.csv \
        --model_id ${folder_data}_${seq_len}'_'${pred_len} \
        --model $model_name \
        --data $folder_data \
        --features $feature_type \
        --target $target \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --embed_type $embed_type \
        --moving_avg $moving_avg \
        --e_layers $e_layer \
        --d_model $d_model \
        --d_target   1 \
        --d_layers   1 \
        --factor     3 \
        --enc_in $num_features \
        --dec_in $num_features_overlap \
        --c_out $num_features \
        --dropout 0.01 \
        --des 'Exp' \
        --loss 'l1' \
        --scheduler 'ReduceLROnPlateau' \
        --train_epochs 100 \
        --batch_size $batch_size \
        --learning_rate 0.001 \
        --is_noscaley \
        --itr 1\
        --option_Ihat1 $option_Ihat1\
        --m2_name $m2_name\
        --checkpoints $checkpoints
done

