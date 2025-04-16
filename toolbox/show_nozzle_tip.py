import pandas as pd
import os

# === 使用者設定 ===
csv_path = r'C:\led_code\MV_AI_mid_ssd\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv'
image_root = r'C:\led_code\MV_AI_mid_ssd\data\plt\nozzlp'  # 不包含 caxton_dataset

# === 讀取 CSV 檔案 ===
df = pd.read_csv(csv_path)

# === 過濾 img_path 為 print0~print4 的資料 ===
df = df[df['img_path'].str.contains(r'caxton_dataset/print[0-4]/')]

# === 隨機抽樣 20 筆資料 ===
sample_df = df[['img_path', 'nozzle_tip_x', 'nozzle_tip_y', 'img_mean', 'img_std']].sample(n=2000, random_state=42)

# === 建立完整路徑欄位 ===
sample_df['full_image_path'] = sample_df['img_path'].apply(lambda x: os.path.join(image_root, x.replace('/', os.sep)))

# === 顯示抽樣結果 ===
print("\n【抽樣影像資訊（來自 print0~print4，共20筆）】")
print(sample_df[['full_image_path', 'nozzle_tip_x', 'nozzle_tip_y', 'img_mean', 'img_std']].to_string(index=False))




import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_and_resize(image_path, center_x, center_y, crop_size=320, target_size=224):
    """
    從影像中以 (center_x, center_y) 為中心裁切出 crop_size*crop_size 的區域並縮放成 target_size。
    若裁切區域超出邊界會自動填補黑邊。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    h, w, _ = image.shape
    half_crop = crop_size // 2

    # 計算裁切邊界
    x1 = int(center_x) - half_crop
    y1 = int(center_y) - half_crop
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # 建立空白畫布進行貼齊（避免邊界溢出）
    canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    # 計算原圖中要貼到畫布上的區域
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = max(0, -x1)
    dst_y1 = max(0, -y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # 縮放為目標大小
    resized = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized


# ✅ 示範裁切並顯示前3張圖
for idx, row in sample_df.head(2000).iterrows():
    img_path = row['full_image_path']
    cx, cy = row['nozzle_tip_x'], row['nozzle_tip_y']
    print(img_path)

    result = crop_and_resize(img_path, cx, cy)

    # 顯示
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Cropped + Resized: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.show()
