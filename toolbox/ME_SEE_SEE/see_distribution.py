import pandas as pd
import matplotlib.pyplot as plt

# 設定 CSV 路徑
csv_path = r"C:\led_code\MV_AI_mid_ssd\toolbox\temp\caxton_dataset_filtered_no_outliers_img_info.csv"

# 只讀取需要的類別欄位
cols_to_read = ['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']
df = pd.read_csv(csv_path, usecols=cols_to_read)

# 繪圖設定
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

# 每個類別欄位畫一張直方圖
for i, col in enumerate(cols_to_read):
    df[col].value_counts().sort_index().plot(kind='bar', ax=axs[i])
    axs[i].set_title(f'Distribution of {col}')
    axs[i].set_xlabel('Class')
    axs[i].set_ylabel('Count')

plt.tight_layout()
plt.show()
