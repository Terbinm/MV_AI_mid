"""
資料預處理模組
用於3D列印錯誤偵測資料集的預處理操作
"""

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from utils.font_utils import chinese_font
from utils.dataset import DatasetProcessor


def analyze_csv_data(csv_path, output_dir=None):
    """
    分析CSV資料集的基本統計特性

    Args:
        csv_path: CSV檔案路徑
        output_dir: 輸出圖表目錄

    Returns:
        stats_dict: 統計資訊字典
    """
    print(f"分析CSV資料: {csv_path}")

    # 讀取CSV檔案
    df = pd.read_csv(csv_path)

    # 基本統計資訊
    print(f"資料總筆數: {len(df)}")
    print(f"欄位數量: {len(df.columns)}")
    print(f"欄位名稱: {', '.join(df.columns)}")

    # 計算參數分佈
    param_cols = ['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']
    stats_dict = {'total_rows': len(df)}

    for col in param_cols:
        if col in df.columns:
            # 參數分佈計數
            value_counts = df[col].value_counts().sort_index()
            stats_dict[col] = value_counts.to_dict()

            print(f"\n{col} 分佈:")
            for val, count in value_counts.items():
                print(f"  值={val}: {count} 筆 ({count / len(df) * 100:.2f}%)")

            # 繪製分佈圖
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.figure(figsize=(10, 6))
                value_counts.plot(kind='bar')
                plt.title(f'{col} 分佈', fontproperties=chinese_font)
                plt.xlabel('參數值', fontproperties=chinese_font)
                plt.ylabel('筆數', fontproperties=chinese_font)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
                plt.close()

    # 分析print_id分佈
    if 'print_id' in df.columns:
        print_ids = df['print_id'].unique()
        stats_dict['unique_print_ids'] = len(print_ids)
        stats_dict['print_id_counts'] = df['print_id'].value_counts().to_dict()

        print(f"\n獨特列印任務數量: {len(print_ids)}")
        print(f"每個列印任務的平均影像數: {len(df) / len(print_ids):.2f}")

    # 分析影像統計資訊 (如果有)
    if 'img_mean' in df.columns and 'img_std' in df.columns:
        img_mean_avg = df['img_mean'].mean()
        img_std_avg = df['img_std'].mean()

        stats_dict['img_mean_avg'] = img_mean_avg
        stats_dict['img_std_avg'] = img_std_avg

        print(f"\n影像平均值的平均: {img_mean_avg:.4f}")
        print(f"影像標準差的平均: {img_std_avg:.4f}")

        # 繪製影像統計分佈
        if output_dir:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(df['img_mean'], bins=50)
            plt.title('影像平均值分佈', fontproperties=chinese_font)
            plt.xlabel('平均值', fontproperties=chinese_font)
            plt.ylabel('頻率', fontproperties=chinese_font)

            plt.subplot(1, 2, 2)
            plt.hist(df['img_std'], bins=50)
            plt.title('影像標準差分佈', fontproperties=chinese_font)
            plt.xlabel('標準差', fontproperties=chinese_font)
            plt.ylabel('頻率', fontproperties=chinese_font)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'image_stats_distribution.png'))
            plt.close()

    # 創建錯誤標籤
    df['error_label'] = ((df['flow_rate_class'] != 1) |
                         (df['feed_rate_class'] != 1) |
                         (df['z_offset_class'] != 1) |
                         (df['hotend_class'] != 1)).astype(int)

    # 統計錯誤標籤分布
    error_count = df['error_label'].sum()
    normal_count = len(df) - error_count
    error_rate = error_count / len(df)

    stats_dict['error_count'] = error_count
    stats_dict['normal_count'] = normal_count
    stats_dict['error_rate'] = error_rate

    print(f"\n錯誤統計:")
    print(f"  正常樣本: {normal_count} ({normal_count / len(df) * 100:.2f}%)")
    print(f"  錯誤樣本: {error_count} ({error_count / len(df) * 100:.2f}%)")

    # 繪製錯誤標籤分佈圖
    if output_dir:
        plt.figure(figsize=(8, 6))
        plt.pie([normal_count, error_count],
                labels=['正常', '錯誤'],
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'])
        plt.title('錯誤與正常樣本分佈', fontproperties=chinese_font)
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        plt.close()

    # 分析不同錯誤類型的組合
    df['error_type'] = ''

    # 創建錯誤類型標記
    for i, row in df.iterrows():
        error_types = []
        if row['flow_rate_class'] != 1:
            error_types.append('flow')
        if row['feed_rate_class'] != 1:
            error_types.append('feed')
        if row['z_offset_class'] != 1:
            error_types.append('z_offset')
        if row['hotend_class'] != 1:
            error_types.append('hotend')

        df.at[i, 'error_type'] = '_'.join(error_types) if error_types else 'normal'

    error_type_counts = df['error_type'].value_counts()
    stats_dict['error_type_counts'] = error_type_counts.to_dict()

    print("\n錯誤類型分佈:")
    for error_type, count in error_type_counts.items():
        print(f"  {error_type}: {count} ({count / len(df) * 100:.2f}%)")

    # 繪製錯誤類型分佈
    if output_dir and len(error_type_counts) > 1:
        plt.figure(figsize=(14, 8))
        error_type_counts.plot(kind='bar')
        plt.title('不同錯誤類型分佈', fontproperties=chinese_font)
        plt.xlabel('錯誤類型', fontproperties=chinese_font)
        plt.ylabel('筆數', fontproperties=chinese_font)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_type_distribution.png'))
        plt.close()

    return stats_dict


def balance_dataset(csv_path, output_csv_path, max_samples=None, balance_ratio=0.5, strategy='hybrid', random_state=42):
    """
    平衡資料集，處理正常和錯誤樣本的不平衡問題

    Args:
        csv_path: 原始CSV檔案路徑
        output_csv_path: 輸出CSV檔案路徑
        max_samples: 最大樣本數量，None表示不限制
        balance_ratio: 錯誤樣本佔總樣本的比例 (0.0-1.0)
        strategy: 平衡策略 ('undersample', 'oversample', 'hybrid')
        random_state: 隨機種子

    Returns:
        平衡後的DataFrame
    """
    print(f"平衡資料集: {csv_path} -> {output_csv_path}")
    print(f"使用策略: {strategy}, 平衡比例: {balance_ratio}, 最大樣本數: {max_samples}")

    # 讀取原始CSV
    df = pd.read_csv(csv_path)
    print(f"原始資料集: {len(df)} 筆")

    # 創建錯誤標籤
    if 'error_label' not in df.columns:
        df['error_label'] = ((df['flow_rate_class'] != 1) |
                             (df['feed_rate_class'] != 1) |
                             (df['z_offset_class'] != 1) |
                             (df['hotend_class'] != 1)).astype(int)

    # 分離正常和錯誤樣本
    normal_df = df[df['error_label'] == 0]
    error_df = df[df['error_label'] == 1]

    normal_count = len(normal_df)
    error_count = len(error_df)

    print(f"原始分布:")
    print(f"  - 正常樣本: {normal_count} ({normal_count / len(df) * 100:.2f}%)")
    print(f"  - 錯誤樣本: {error_count} ({error_count / len(df) * 100:.2f}%)")

    # 設置隨機種子
    np.random.seed(random_state)

    # 根據不同策略平衡資料集
    if strategy == 'undersample':
        # 下採樣策略：基於少數類別(正常樣本)的數量，對多數類別進行下採樣
        target_error_count = int(normal_count / (1 - balance_ratio) * balance_ratio)
        target_error_count = min(target_error_count, error_count)
        target_normal_count = normal_count

        # 最大樣本數限制
        if max_samples and target_normal_count + target_error_count > max_samples:
            ratio = max_samples / (target_normal_count + target_error_count)
            target_normal_count = int(target_normal_count * ratio)
            target_error_count = int(target_error_count * ratio)

        # 抽樣
        if target_normal_count < normal_count:
            sampled_normal = normal_df.sample(target_normal_count, random_state=random_state)
        else:
            sampled_normal = normal_df

        sampled_error = error_df.sample(target_error_count, random_state=random_state)

    elif strategy == 'oversample':
        # 過採樣策略：基於多數類別(錯誤樣本)的數量，對少數類別進行過採樣
        target_normal_count = int(error_count / balance_ratio * (1 - balance_ratio))
        target_error_count = error_count

        # 最大樣本數限制
        if max_samples and target_normal_count + target_error_count > max_samples:
            ratio = max_samples / (target_normal_count + target_error_count)
            target_normal_count = int(target_normal_count * ratio)
            target_error_count = int(target_error_count * ratio)

        # 抽樣 - 對少數類進行放回抽樣
        if target_normal_count > normal_count:
            # 過採樣：放回抽樣
            indices = np.random.choice(normal_df.index, size=target_normal_count, replace=True)
            sampled_normal = normal_df.loc[indices].reset_index(drop=True)
        else:
            sampled_normal = normal_df.sample(target_normal_count, random_state=random_state)

        sampled_error = error_df.sample(target_error_count, random_state=random_state)

    elif strategy == 'hybrid':
        # 混合策略：同時下採樣多數類和過採樣少數類
        # 計算理想的平衡後總樣本數
        if max_samples is None:
            # 預設使用現有樣本數的一半
            target_total = len(df) // 2
        else:
            target_total = max_samples

        # 計算平衡後的錯誤和正常樣本數
        target_error_count = int(target_total * balance_ratio)
        target_normal_count = target_total - target_error_count

        # 抽樣
        if target_normal_count <= normal_count:
            # 正常樣本不需要過採樣
            sampled_normal = normal_df.sample(target_normal_count, random_state=random_state)
        else:
            # 正常樣本需要過採樣
            indices = np.random.choice(normal_df.index, size=target_normal_count, replace=True)
            sampled_normal = normal_df.loc[indices].reset_index(drop=True)

        if target_error_count <= error_count:
            # 錯誤樣本不需要過採樣
            sampled_error = error_df.sample(target_error_count, random_state=random_state)
        else:
            # 錯誤樣本需要過採樣
            indices = np.random.choice(error_df.index, size=target_error_count, replace=True)
            sampled_error = error_df.loc[indices].reset_index(drop=True)

    else:
        raise ValueError(f"不支援的平衡策略: {strategy}")

    # 合併並打亂順序
    balanced_df = pd.concat([sampled_normal, sampled_error])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 分析平衡後的分布
    normal_count = (balanced_df['error_label'] == 0).sum()
    error_count = (balanced_df['error_label'] == 1).sum()

    print(f"平衡後分布 (策略: {strategy}):")
    print(f"  - 正常樣本: {normal_count} ({normal_count / len(balanced_df) * 100:.2f}%)")
    print(f"  - 錯誤樣本: {error_count} ({error_count / len(balanced_df) * 100:.2f}%)")
    print(f"  - 總樣本數: {len(balanced_df)}")

    # 保存平衡後的CSV
    balanced_df.to_csv(output_csv_path, index=False)
    print(f"平衡後的資料集已保存至: {output_csv_path}")

    return balanced_df


def check_image_paths(csv_path, image_dir, verbose=True):
    """
    檢查CSV中的影像路徑是否都存在

    Args:
        csv_path: CSV檔案路徑
        image_dir: 影像目錄根路徑
        verbose: 是否輸出詳細資訊

    Returns:
        (valid_count, missing_count): 有效和缺失的影像數量
    """
    print(f"檢查影像路徑: {csv_path}")

    # 讀取CSV
    df = pd.read_csv(csv_path)

    if 'img_path' not in df.columns:
        raise ValueError("CSV中找不到 'img_path' 欄位")

    valid_count = 0
    missing_count = 0
    missing_paths = []

    # 檢查每個影像路徑
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="檢查影像"):
        img_path = os.path.join(image_dir, row['img_path'])

        if os.path.isfile(img_path):
            valid_count += 1
        else:
            missing_count += 1
            missing_paths.append(row['img_path'])

            if verbose and missing_count <= 10:  # 只顯示前10個缺失的影像
                print(f"找不到影像: {img_path}")

    print(f"檢查完成: 共 {valid_count} 個有效影像, {missing_count} 個缺失影像")

    if missing_count > 0 and verbose:
        if missing_count > 10:
            print(f"缺失影像過多，僅顯示前10個。總共缺失 {missing_count} 個影像。")
        print(f"請確認影像目錄 {image_dir} 是否正確")

    return valid_count, missing_count, missing_paths


def sample_and_explore_images(csv_path, image_dir, output_dir, num_samples=5, random_seed=42):
    """
    從資料集中抽樣並探索影像特性

    Args:
        csv_path: CSV檔案路徑
        image_dir: 影像目錄根路徑
        output_dir: 輸出目錄
        num_samples: 抽樣數量
        random_seed: 隨機種子
    """
    print(f"抽樣並探索影像: {csv_path}")

    # 讀取CSV
    df = pd.read_csv(csv_path)

    # 創建錯誤標籤(如果不存在)
    if 'error_label' not in df.columns:
        df['error_label'] = ((df['flow_rate_class'] != 1) |
                             (df['feed_rate_class'] != 1) |
                             (df['z_offset_class'] != 1) |
                             (df['hotend_class'] != 1)).astype(int)

    # 分別從正常和錯誤樣本中抽樣
    normal_df = df[df['error_label'] == 0]
    error_df = df[df['error_label'] == 1]

    np.random.seed(random_seed)

    # 確保有足夠的樣本可抽
    normal_samples = min(num_samples, len(normal_df))
    error_samples = min(num_samples, len(error_df))

    # 隨機抽樣
    sampled_normal = normal_df.sample(normal_samples)
    sampled_error = error_df.sample(error_samples)

    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 處理正常樣本
    print("處理正常樣本...")
    for idx, row in sampled_normal.iterrows():
        img_path = os.path.join(image_dir, row['img_path'])
        if not os.path.isfile(img_path):
            print(f"警告: 找不到影像 {img_path}")
            continue

        # 讀取影像
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 無法讀取影像 {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 繪製影像資訊
        plt.figure(figsize=(15, 10))

        # 原始影像
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title(f"原始影像 (正常)", fontproperties=chinese_font)
        plt.axis('off')

        # 影像直方圖
        plt.subplot(2, 2, 2)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title('RGB直方圖', fontproperties=chinese_font)
        plt.xlim([0, 256])

        # 噴嘴區域(如果有座標)
        if 'nozzle_tip_x' in row and 'nozzle_tip_y' in row:
            nozzle_x = int(row['nozzle_tip_x'])
            nozzle_y = int(row['nozzle_tip_y'])

            # 噴嘴附近區域
            crop_size = 320
            x1 = max(0, nozzle_x - crop_size // 2)
            y1 = max(0, nozzle_y - crop_size // 2)
            x2 = min(img.shape[1], x1 + crop_size)
            y2 = min(img.shape[0], y1 + crop_size)

            # 調整起始點
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)

            cropped = img[y1:y2, x1:x2]

            plt.subplot(2, 2, 3)
            plt.imshow(cropped)
            plt.title(f"噴嘴區域 ({crop_size}x{crop_size})", fontproperties=chinese_font)
            plt.axis('off')

            # 標示噴嘴位置
            marked = img.copy()
            cv2.circle(marked, (nozzle_x, nozzle_y), 15, (255, 0, 0), 3)

            plt.subplot(2, 2, 4)
            plt.imshow(marked)
            plt.title("噴嘴位置標記", fontproperties=chinese_font)
            plt.axis('off')

        # 保存圖表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"normal_sample_{idx}.png"))
        plt.close()

    # 處理錯誤樣本
    print("處理錯誤樣本...")
    for idx, row in sampled_error.iterrows():
        img_path = os.path.join(image_dir, row['img_path'])
        if not os.path.isfile(img_path):
            print(f"警告: 找不到影像 {img_path}")
            continue

        # 讀取影像
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 無法讀取影像 {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 獲取錯誤參數信息
        error_info = []
        if row['flow_rate_class'] != 1:
            status = "低" if row['flow_rate_class'] < 1 else "高"
            error_info.append(f"流量={status} ({row['flow_rate']})")
        if row['feed_rate_class'] != 1:
            status = "低" if row['feed_rate_class'] < 1 else "高"
            error_info.append(f"進給={status} ({row['feed_rate']})")
        if row['z_offset_class'] != 1:
            status = "低" if row['z_offset_class'] < 1 else "高"
            error_info.append(f"Z偏移={status} ({row['z_offset']})")
        if row['hotend_class'] != 1:
            status = "低" if row['hotend_class'] < 1 else "高"
            error_info.append(f"噴頭溫度={status} ({row['hotend']})")

        error_text = ", ".join(error_info)

        # 繪製影像資訊
        plt.figure(figsize=(15, 10))

        # 原始影像
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title(f"原始影像 (錯誤)\n{error_text}", fontproperties=chinese_font)
        plt.axis('off')

        # 影像直方圖
        plt.subplot(2, 2, 2)
        for i, color in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title('RGB直方圖', fontproperties=chinese_font)
        plt.xlim([0, 256])

        # 噴嘴區域(如果有座標)
        if 'nozzle_tip_x' in row and 'nozzle_tip_y' in row:
            nozzle_x = int(row['nozzle_tip_x'])
            nozzle_y = int(row['nozzle_tip_y'])

            # 噴嘴附近區域
            crop_size = 320
            x1 = max(0, nozzle_x - crop_size // 2)
            y1 = max(0, nozzle_y - crop_size // 2)
            x2 = min(img.shape[1], x1 + crop_size)
            y2 = min(img.shape[0], y1 + crop_size)

            # 調整起始點
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)

            cropped = img[y1:y2, x1:x2]

            plt.subplot(2, 2, 3)
            plt.imshow(cropped)
            plt.title(f"噴嘴區域 ({crop_size}x{crop_size})", fontproperties=chinese_font)
            plt.axis('off')

            # 創建全紅色遮罩，表示整個區域有錯誤
            mask = np.zeros_like(cropped)
            mask[:, :, 0] = 255  # 紅色通道設為最大

            # 紅色半透明遮罩疊加在裁剪影像上
            overlay = cv2.addWeighted(cropped, 0.7, mask, 0.3, 0)

            plt.subplot(2, 2, 4)
            plt.imshow(overlay)
            plt.title("錯誤區域遮罩", fontproperties=chinese_font)
            plt.axis('off')

        # 保存圖表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"error_sample_{idx}.png"))
        plt.close()

    print(f"已保存 {normal_samples} 個正常樣本和 {error_samples} 個錯誤樣本到 {output_dir}")


def cleanup_processed_data(config, confirm=True):
    """
    清理先前處理的資料

    Args:
        config: 配置對象
        confirm: 是否需要確認
    """
    # 確認刪除
    if confirm:
        response = input("確定要刪除所有處理後的資料嗎？這將刪除所有預處理資料。 [y/N]: ")
        if response.lower() != 'y':
            print("已取消刪除操作")
            return

    # 刪除處理後的資料目錄
    processed_dir = os.path.join(config.DATA_DIR, 'processed')
    if os.path.exists(processed_dir):
        print("正在刪除處理後的資料...")
        shutil.rmtree(processed_dir)
        print(f"已成功刪除目錄: {processed_dir}")
    else:
        print(f"找不到處理後的資料目錄: {processed_dir}")


def preprocess_all_data(config):
    """
    執行完整的資料預處理流程

    Args:
        config: 配置物件
    """
    print(f"開始資料預處理流程...")

    # 創建必要的目錄
    os.makedirs(os.path.join(config.DATA_DIR, 'processed', 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'processed', 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'processed', 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'processed', 'val', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'plt'), exist_ok=True)

    # 分析CSV資料
    csv_path = os.path.join(config.DATA_DIR, 'limited_data_size', config.CSV_FILE)
    stats = analyze_csv_data(csv_path, os.path.join(config.DATA_DIR, 'plt'))

    # 檢查影像路徑
    valid_count, missing_count, _ = check_image_paths(csv_path, config.DATA_DIR)

    if valid_count == 0:
        raise ValueError("找不到任何有效影像，請檢查影像路徑")

    # 抽樣並探索影像特性
    sample_and_explore_images(
        csv_path,
        config.DATA_DIR,
        os.path.join(config.DATA_DIR, 'plt', 'sample_images'),
        num_samples=5
    )

    # 平衡資料集
    balanced_csv_path = os.path.join(config.DATA_DIR, 'processed', 'balanced_dataset.csv')

    print(f"創建平衡樣本: {csv_path} -> {balanced_csv_path}")

    # 根據配置選擇是否平衡資料集
    if config.BALANCE_DATASET:
        balanced_df = balance_dataset(
            csv_path=csv_path,
            output_csv_path=balanced_csv_path,
            max_samples=config.MAX_SAMPLES,
            balance_ratio=config.BALANCE_RATIO,
            strategy=config.BALANCE_STRATEGY
        )
        working_csv_path = balanced_csv_path
    else:
        # 直接使用原始CSV
        print("跳過資料平衡，使用原始資料集")
        shutil.copy(csv_path, balanced_csv_path)
        working_csv_path = balanced_csv_path
        balanced_df = pd.read_csv(working_csv_path)

    # 創建處理器對象
    processor = DatasetProcessor(
        csv_path=working_csv_path,
        image_dir=config.DATA_DIR,
        config=config
    )

    # 分割訓練和驗證集
    print("分割訓練集和驗證集...")
    train_indices, val_indices = processor.split_train_val(val_ratio=config.VAL_RATIO)

    # 處理訓練集影像
    print("預處理訓練集影像...")
    processor.preprocess_and_save_images(
        indices=train_indices,
        target_dir=os.path.join(config.DATA_DIR, 'processed', 'train'),
        crop_size=config.CROP_SIZE,
        resize=config.RESIZE
    )

    # 處理驗證集影像
    print("預處理驗證集影像...")
    processor.preprocess_and_save_images(
        indices=val_indices,
        target_dir=os.path.join(config.DATA_DIR, 'processed', 'val'),
        crop_size=config.CROP_SIZE,
        resize=config.RESIZE
    )

    print("資料預處理完成！")
    return processor