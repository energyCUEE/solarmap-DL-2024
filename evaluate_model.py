import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pdb

model_name = "RLSTM"
dataset = "CUEE_PMAPS_NIGHT"
moving_average = 4  # mv
mode = "MS"  # ft
enc_in = 11  # enc
seq_length = 4  # sl
# ll0
pred_length = 1  # pl
d_model = 50  # dm
# nh8
e_layers = 5  # el
# dl1
# df2048
fc = 1  # fc
# ebtimeF
# dtTrue
# dp0p10
# Exp
# l1loss
# 0
main_folder_path = "results"

folder_list = [
    "%s_%s_mv%d_ft%s_enc%d_sl%d_ll0_pl%d_dm%d_nh8_el%d_dl1_df2048_fc%d_etype0_ebtimeF_dtTrue_dp0p10_Exp_l1loss_0"
    % (model_name, dataset, moving_average, mode, enc_in, seq_length, pred_length, d_model, e_layers, fc),
]

print(folder_list)
stats_df = pd.read_csv(os.path.join(main_folder_path, folder_list[0], 'stats_mae_mse.csv'))

df = pd.read_csv(os.path.join(main_folder_path, folder_list[0], 'result_dict.csv'))
df.drop(columns={'inputx', 'preds', 'trues'}, inplace=True)

df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
df['datetime'] = df['datetime'].dt.tz_localize('UTC')
df['datetime'] = df['datetime'].dt.tz_convert('Asia/Bangkok')
df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['hour'] = pd.to_datetime(df['datetime']).dt.hour

df.rename(columns={'trues_rev': 'I', 'preds_rev': 'Ihat'}, inplace=True)
pdb.set_trace()
df['sky_condition_kbar'] = df['average_k'].apply(
    lambda x: 'cloudy' if x < 0.3 else ('partly_cloudy' if x < 0.6 else 'clear'))

sky_condition_mae = df.groupby('sky_condition_kbar')[['I', 'Ihat']].apply(
    lambda x: mean_absolute_error(x['I'], x['Ihat'])).reset_index(name='MAE')
sky_condition_rmse = df.groupby('sky_condition_kbar')[['I', 'Ihat']].apply(
    lambda x: np.sqrt(mean_squared_error(x['I'], x['Ihat']))).reset_index(name='RMSE')

overall_mae = mean_absolute_error(df['I'], df['Ihat'])
overall_rmse = np.sqrt(mean_squared_error(df['I'], df['Ihat']))
print(f'Overall MAE: {overall_mae:.2f}')
print(f'Overall RMSE: {overall_rmse:.2f}')

print('MAE by sky_condition')
print(sky_condition_mae)

print('\nRMSE by sky_condition')
print(sky_condition_rmse)

stats_data = pd.concat([stats_df, sky_condition_mae, sky_condition_rmse], axis=0)
stats_data.to_csv(os.path.join(main_folder_path, folder_list[0], 'stats_mae_mbe_skycondition.csv'))

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
sky_condition_names = ['Clear sky', 'Partly cloudy sky', 'Cloudy sky']

# Define three different color sets for each subplot
color_sets = [['#86A789', '#D2E3C8'], ['#22668D', '#8ECDDD'], ['#2D3250', '#7077A1']]

for i, condition in enumerate(['clear', 'partly_cloudy', 'cloudy']):
    filter_test_df = df[df['sky_condition_kbar'] == condition]
    mae = filter_test_df.groupby('hour')[['I', 'Ihat']].apply(
        lambda x: mean_absolute_error(x['I'], x['Ihat'])).reset_index(name='MAE')
    mbe = filter_test_df.groupby('hour')[['I', 'Ihat']].apply(lambda x: x['Ihat'].mean() - x['I'].mean()).reset_index(
        name='MBE')

    # Use different color sets for each subplot
    bar_colors_mae = [color_sets[i][0] if 10 <= hour <= 15 else color_sets[i][1] for hour in mae['hour']]
    bar_colors_mbe = [color_sets[i][0] if 10 <= hour <= 15 else color_sets[i][1] for hour in mbe['hour']]

    axes[0, i].bar(mae['hour'], mae['MAE'], color=bar_colors_mae)
    axes[0, i].set_title(f'{sky_condition_names[i]} (n={len(filter_test_df)})', fontsize=20)
    axes[0, i].set_xlabel('Hour', fontsize=20)
    axes[0, i].set_ylabel('MAE [W/sqm]', fontsize=20)

    axes[1, i].bar(mbe['hour'], mbe['MBE'], color=bar_colors_mbe)
    axes[1, i].set_xlabel('Hour', fontsize=20)
    axes[1, i].set_ylabel('MBE [W/sqm]', fontsize=20)

fig.tight_layout()
fig.subplots_adjust(top=0.85)
plt.savefig(os.path.join(main_folder_path, folder_list[0], 'test_metric.png'))
plt.show()