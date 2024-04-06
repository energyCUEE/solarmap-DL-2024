import csv
import pandas as pd
import pdb
csv_filename = "EE Station 1-20230101-20231231.csv"
read_data    = pd.read_csv(csv_filename) 
read_data    = read_data.get(["Datetime",  "Irradiance_30 (W/m2)"]).copy()
read_data   = read_data.dropna()
read_data.rename(columns = {'Irradiance_30 (W/m2)':'I'}, inplace = True) 
print(read_data)
pdb.set_trace()
read_data.to_csv("EE-Station-20230101-20231231_rmNan.csv", index=False)

