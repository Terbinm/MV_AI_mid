import pandas as pd
import matplotlib.pyplot as plt

# CSV 路徑
csv_path = r"C:\led_code\MV_AI_mid_ssd\toolbox\temp\caxton_dataset_filtered_no_outliers_img_info.csv"

# 讀取必要欄位
cols = ['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']
df = pd.read_csv(csv_path, usecols=cols)

# 計算每一列有幾個欄位是異常（class ≠ 1）
df['num_abnormal'] = df.apply(lambda row: sum(col != 1 for col in row), axis=1)

# 統計異常數量（0～4）的分布
distribution = df['num_abnormal'].value_counts().sort_index()

# 繪製直方圖
plt.figure(figsize=(8, 5))
distribution.plot(kind='bar')
plt.title('Distribution of Abnormal Class Count per Row')
plt.xlabel('Number of Abnormal Classes (out of 4)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
