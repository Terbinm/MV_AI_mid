import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 路徑
csv_path = r"C:\led_code\MV_AI_mid_ssd\toolbox\temp\caxton_dataset_filtered_no_outliers_img_info.csv"

# 只讀需要的欄位以節省記憶體
cols = ["flow_rate_class", "feed_rate_class", "z_offset_class", "hotend_class"]
df = pd.read_csv(csv_path, usecols=cols)

# 所有要畫的混淆矩陣欄位對
pairs = [
    ("flow_rate_class", "feed_rate_class"),
    ("flow_rate_class", "z_offset_class"),
    ("flow_rate_class", "hotend_class"),
    ("feed_rate_class", "z_offset_class"),
    ("feed_rate_class", "hotend_class"),
    ("z_offset_class", "hotend_class"),
]

# 畫圖設定
plt.figure(figsize=(16, 10))
plt.suptitle("Cross Distribution of Classification Fields", fontsize=16)

for i, (col1, col2) in enumerate(pairs):
    # 混淆矩陣統計
    cm = pd.crosstab(df[col1], df[col2])

    # 畫子圖
    plt.subplot(2, 3, i + 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.title(f"{col1} vs {col2}")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
