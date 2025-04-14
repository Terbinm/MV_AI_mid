import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import time
import numpy as np

start_time = time.time()
# 設定資料夾路徑
input_dir = r"D:\led\MV_AI_mid\data\raw"
output_dir = r"D:\led\MV_AI_mid\data\limited_data_size"
plot_dir = r"D:\led\MV_AI_mid\data\plt"

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# 設定最大的print編號 (這裡可以根據需要修改)
max_print_num = 29

# 要處理的CSV檔案清單
csv_files = [
    "caxton_dataset_filtered.csv",
    "caxton_dataset_filtered_no_outliers.csv",
    "caxton_dataset_filtered_no_outliers_img_info.csv",
    "caxton_dataset_final.csv",
    "caxton_dataset_full.csv"
]

# 輸出的CSV檔案清單
out_files = [
    "caxton_dataset_filtered.csv",
    "caxton_dataset_filtered_no_outliers.csv",
    "caxton_dataset_filtered_no_outliers_img_info.csv",
    "caxton_dataset_final.csv",
    "caxton_dataset_full.csv"
]
# out_files = [
#     "limit_id_dataset_filtered.csv",
#     "limit_id_dataset_filtered_no_outliers.csv",
#     "limit_id_dataset_filtered_no_outliers_img_info.csv",
#     "limit_id_dataset_final.csv",
#     "limit_id_dataset_full.csv"
# ]

# 針對每個CSV檔案進行處理
for i, csv_file in enumerate(csv_files):
    input_csv_path = os.path.join(input_dir, csv_file)
    # 使用對應的out_files檔名
    output_csv_path = os.path.join(output_dir, out_files[i])
    # 繪圖的檔名仍然基於原始檔案名稱，但可以根據需要修改為使用out_files的名稱
    output_plot_path = os.path.join(plot_dir, f"{os.path.splitext(out_files[i])[0]}_distribution.png")

    print(f"\n處理檔案: {csv_file}")
    print(f"輸入路徑: {input_csv_path}")
    print(f"輸出路徑: {output_csv_path}")
    print(f"輸出圖表路徑: {output_plot_path}")

    # 檢查檔案是否存在
    if not os.path.exists(input_csv_path):
        print(f"錯誤: 檔案 {input_csv_path} 不存在，跳過處理")
        continue

    try:
        # 讀取CSV檔案
        print(f"正在讀取CSV檔案...")
        df = pd.read_csv(input_csv_path)

        # 顯示原始資料筆數
        print(f"原始資料筆數: {len(df)}")

        # 空間換時間：預先提取所有img_path，減少重複存取
        img_paths = df['img_path'].astype(str).tolist()

        # 使用正則表達式一次性提取所有print編號
        print_pattern = re.compile(r'caxton_dataset/print(\d+)/')
        all_print_numbers = set()

        for img_path in img_paths:
            match = print_pattern.search(img_path)
            if match:
                print_num = int(match.group(1))
                all_print_numbers.add(print_num)

        all_print_numbers = sorted(list(all_print_numbers))

        if not all_print_numbers:
            print(f"警告: 在檔案 {csv_file} 中找不到任何 print 目錄，跳過處理")
            continue

        print(f"發現的print編號: {all_print_numbers}")
        print(f"最大的print編號: {max(all_print_numbers)}")

        # 預先建立一個dictionary將每一筆資料對應到其print編號，提高後續計算效率
        data_print_map = {}
        for i, img_path in enumerate(img_paths):
            match = print_pattern.search(img_path)
            if match:
                print_num = int(match.group(1))
                data_print_map[i] = print_num

        # 計算原始資料中各print編號的數量
        original_counts = {}
        for i in all_print_numbers:
            # 使用提前計算的映射來統計
            count = sum(1 for idx, print_num in data_print_map.items() if print_num == i)
            original_counts[f'print{i}'] = count
            print(f"原始資料中 print{i}/ 目錄資料筆數: {count}")

        # 正確設定篩選條件：使用一個合適的方法篩選資料
        # 創建一個篩選條件的布林陣列
        filter_mask = [data_print_map.get(i, -1) <= max_print_num for i in range(len(df))]

        # 過濾資料
        filtered_df = df[filter_mask]

        # 顯示過濾後的資料筆數
        print(f"過濾後資料筆數: {len(filtered_df)}")

        # 儲存過濾後的資料
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"已將過濾後的資料儲存至: {output_csv_path}")

        # 計算過濾後的資料中各print編號的數量
        filtered_counts = {}
        # 只考慮小於等於max_print_num的編號
        for i in range(max_print_num + 1):
            if i in all_print_numbers:  # 確認這個編號在原始資料中存在
                filtered_count = sum(1 for idx, print_num in data_print_map.items()
                                     if print_num == i and idx in filtered_df.index)
                filtered_counts[f'print{i}'] = filtered_count
                print(f"過濾後資料中 print{i}/ 目錄資料筆數: {filtered_count}")

        # 繪製直方圖比較
        plt.figure(figsize=(14, 8))

        # 獲取所有print編號
        all_prints = [f'print{i}' for i in all_print_numbers]
        x = np.arange(len(all_prints))
        width = 0.35

        # 準備繪製資料
        original_values = []
        filtered_values = []

        for print_dir in all_prints:
            original_values.append(original_counts.get(print_dir, 0))
            filtered_values.append(filtered_counts.get(print_dir, 0))

        # 設定y軸的上限為最大值的1.2倍，確保有足夠空間顯示標籤
        max_value = max(original_values) * 1.2

        # 繪製原始資料的長條圖
        plt.bar(x - width / 2, original_values, width, label='raw')

        # 繪製過濾後資料的長條圖
        plt.bar(x + width / 2, filtered_values, width, label=f'filtered (print0-{max_print_num})')

        # 設定圖表
        plt.xlabel('PrintID')
        plt.ylabel('data size')
        plt.title(f'{os.path.splitext(out_files[i])[0]} - raw VS filtered PrintID')
        plt.xticks(x, all_prints, rotation=45 if len(all_prints) > 10 else 0)
        plt.ylim(0, max_value)  # 設定y軸範圍
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在每個長條上標示數值
        for i, v in enumerate(original_values):
            if v > 0:  # 只有當有值時才標註
                plt.text(i - width / 2, v + max_value * 0.01, f"{v}",
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 只標註過濾後有值的部分
        for i, print_num in enumerate(all_print_numbers):
            if print_num <= max_print_num:
                filtered_value = filtered_counts.get(f'print{print_num}', 0)
                if filtered_value > 0:
                    plt.text(i + width / 2, filtered_value + max_value * 0.01, f"{filtered_value}",
                             ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 儲存圖表
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        print(f"已將分布比較圖儲存至: {output_plot_path}")

        # 關閉圖表以節省記憶體
        plt.close()

    except Exception as e:
        print(f"處理檔案 {csv_file} 時發生錯誤: {str(e)}")

print("\n所有檔案處理完成!")

elapsed_time = time.time() - start_time
print(f"總執行時間: {elapsed_time:.2f} 秒")