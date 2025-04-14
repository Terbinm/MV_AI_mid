import pandas as pd
import numpy as np

# 載入資料
df = pd.read_csv('caxton_dataset_filtered_no_outliers_img_info.csv')

# 顯示前幾筆資料
print(df.head())

# 檢查缺失值
print(df.isnull().sum())
