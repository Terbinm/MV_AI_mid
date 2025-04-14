"""
視覺化工具模組
用於3D列印錯誤偵測結果的視覺化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from tqdm import tqdm
import pandas as pd
from utils.font_utils import chinese_font


def plot_training_history(history, output_path=None):
    """
    繪製訓練歷史曲線

    Args:
        history: 訓練過程歷史記錄對象
        output_path: 輸出路徑
    """
    plt.figure(figsize=(12, 8))

    # 繪製損失曲線
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('損失曲線', fontproperties=chinese_font)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 尋找可用的評估指標
    metrics = [m for m in history.history.keys()
               if not m.startswith('val_') and m != 'loss']

    # 繪製各個評估指標
    for i, metric in enumerate(metrics[:3]):  # 最多繪製3個指標
        plt.subplot(2, 2, i + 2)
        plt.plot(history.history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric} 曲線', fontproperties=chinese_font)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"訓練歷史曲線已保存至 {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_predictions_grid(model, test_images, test_masks, num_samples=6, threshold=0.5, output_path=None):
    """
    視覺化預測結果格網

    Args:
        model: 訓練好的模型
        test_images: 測試影像陣列 [N, H, W, C]
        test_masks: 真實遮罩陣列 [N, H, W, 1]
        num_samples: 要顯示的樣本數量
        threshold: 二值化閾值
        output_path: 輸出路徑
    """
    # 確保樣本數不超過測試集大小
    num_samples = min(num_samples, len(test_images))

    # 隨機選擇樣本
    indices = np.random.choice(len(test_images), num_samples, replace=False)

    # 設置圖表
    plt.figure(figsize=(15, num_samples * 4))

    for i, idx in enumerate(indices):
        # 獲取影像和遮罩
        image = test_images[idx]
        true_mask = test_masks[idx].squeeze()

        # 預測遮罩
        pred_mask = model.predict(np.expand_dims(image, axis=0))[0].squeeze()
        pred_binary = (pred_mask > threshold).astype(np.uint8)

        # 反正規化影像 (如果已正規化)
        if image.max() <= 1.0:
            image_display = (image * 255).astype(np.uint8)
        else:
            image_display = image.astype(np.uint8)

        # 創建疊加效果
        overlay = image_display.copy()
        # 紅色:真實錯誤，綠色:預測錯誤
        overlay[true_mask > 0.5, 0] = overlay[true_mask > 0.5, 0] * 0.5 + 128  # 紅色半透明
        overlay[true_mask > 0.5, 1] = overlay[true_mask > 0.5, 1] * 0.5
        overlay[true_mask > 0.5, 2] = overlay[true_mask > 0.5, 2] * 0.5

        overlay[pred_binary > 0, 0] = overlay[pred_binary > 0, 0] * 0.5
        overlay[pred_binary > 0, 1] = overlay[pred_binary > 0, 1] * 0.5 + 128  # 綠色半透明
        overlay[pred_binary > 0, 2] = overlay[pred_binary > 0, 2] * 0.5

        # 繪製子圖
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(image_display)
        plt.title(f'原始影像 #{idx}', fontproperties=chinese_font)
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(pred_mask, cmap='jet')
        plt.title(f'預測機率', fontproperties=chinese_font)
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(overlay)
        plt.title('紅:真實錯誤，綠:預測錯誤', fontproperties=chinese_font)
        plt.axis('off')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"預測視覺化已保存至 {output_path}")
    else:
        plt.show()

    plt.close()


def create_error_visualization_video(model, video_path, csv_path, image_dir,
                                     output_path, fps=10, threshold=0.5, resize=None):
    """
    創建錯誤偵測視覺化影片

    Args:
        model: 訓練好的模型
        video_path: 輸出影片路徑
        csv_path: CSV檔案路徑
        image_dir: 影像目錄
        output_path: 輸出路徑
        fps: 每秒影格數
        threshold: 二值化閾值
        resize: 調整大小，格式為(width, height)
    """
    print(f"開始創建錯誤偵測視覺化影片...")

    # 讀取CSV
    df = pd.read_csv(csv_path)

    # 排序資料(確保按照時間順序)
    if 'print_id' in df.columns and 'img_num' in df.columns:
        df = df.sort_values(['print_id', 'img_num'])

    # 創建錯誤標籤(如果不存在)
    if 'error_label' not in df.columns:
        df['error_label'] = ((df['flow_rate_class'] != 1) |
                             (df['feed_rate_class'] != 1) |
                             (df['z_offset_class'] != 1) |
                             (df['hotend_class'] != 1)).astype(int)

    # 準備影片寫入器
    input_size = model.input_shape[1:3]  # 模型輸入大小
    if resize is None:
        resize = (input_size[1] * 2, input_size[0] * 2)  # 預設加倍大小

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, resize)

    # 初始化進度條
    progress_bar = tqdm(total=len(df), desc="處理影格")

    # 記錄相關數據以生成圖表
    frame_numbers = []
    error_scores = []
    true_labels = []
    param_values = {'flow_rate': [], 'feed_rate': [], 'z_offset': [], 'hotend': []}

    # 處理每一張影像
    for idx, row in df.iterrows():
        # 讀取影像
        img_path = os.path.join(image_dir, row['img_path'])
        if not os.path.exists(img_path):
            progress_bar.update(1)
            continue

        img = cv2.imread(img_path)
        if img is None:
            progress_bar.update(1)
            continue

        # 裁剪噴嘴區域(如果有座標)
        if 'nozzle_tip_x' in row and 'nozzle_tip_y' in row:
            nozzle_x = int(row['nozzle_tip_x'])
            nozzle_y = int(row['nozzle_tip_y'])

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

            img = img[y1:y2, x1:x2]

        # 調整大小為模型輸入尺寸
        img_resized = cv2.resize(img, (input_size[1], input_size[0]))

        # 正規化影像
        img_norm = img_resized.astype(np.float32) / 255.0

        # 預測
        pred = model.predict(np.expand_dims(img_norm, axis=0))[0].squeeze()

        # 二值化預測結果
        pred_binary = (pred > threshold).astype(np.uint8) * 255

        # 調整大小以便顯示
        pred_display = cv2.resize(pred, (img.shape[1], img.shape[0]))
        pred_binary_display = cv2.resize(pred_binary, (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

        # 創建熱圖
        pred_heatmap = cv2.applyColorMap((pred_display * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 創建疊加效果
        alpha = 0.5
        overlay = cv2.addWeighted(img, 1 - alpha, pred_heatmap, alpha, 0)

        # 添加文字信息
        error_score = np.mean(pred)
        error_status = "錯誤" if error_score > threshold else "正常"
        true_status = "錯誤" if row['error_label'] == 1 else "正常"

        # 準備彙整畫面
        # 左上: 原始影像，右上: 機率熱圖
        # 左下: 二值化結果，右下: 疊加效果
        h, w = img.shape[:2]
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # 放置各個區塊
        canvas[:h, :w] = img  # 原始影像
        canvas[:h, w:] = pred_heatmap  # 機率熱圖

        # 二值化結果 (轉為3通道以便顯示)
        binary_display = np.zeros((h, w, 3), dtype=np.uint8)
        binary_display[..., 0] = pred_binary_display
        binary_display[..., 1] = pred_binary_display
        binary_display[..., 2] = pred_binary_display
        canvas[h:, :w] = binary_display

        canvas[h:, w:] = overlay  # 疊加效果

        # 添加文字說明
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"原始影像", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"錯誤機率熱圖", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"二值化結果", (10, h + 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"疊加效果", (w + 10, h + 30), font, 0.7, (255, 255, 255), 2)

        # 添加參數信息
        info_text = [
            f"影像: {os.path.basename(row['img_path'])}",
            f"預測: {error_status} (得分: {error_score:.3f})",
            f"真實: {true_status}",
            f"流量: {row['flow_rate']}% (類別: {row['flow_rate_class']})",
            f"進給: {row['feed_rate']}% (類別: {row['feed_rate_class']})",
            f"Z偏移: {row['z_offset']} (類別: {row['z_offset_class']})",
            f"噴頭溫度: {row['hotend']}°C (類別: {row['hotend_class']})"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(canvas, text, (10, 2 * h - 120 + i * 20), font, 0.5, (255, 255, 255), 1)

        # 調整最終輸出大小
        canvas_resized = cv2.resize(canvas, resize)

        # 寫入影片
        video_writer.write(canvas_resized)

        # 記錄數據以生成圖表
        frame_numbers.append(len(frame_numbers))
        error_scores.append(error_score)
        true_labels.append(row['error_label'])
        for param in param_values:
            if param in row:
                param_values[param].append(row[param])

        # 更新進度條
        progress_bar.update(1)

    # 關閉影片寫入器
    video_writer.release()
    progress_bar.close()

    print(f"影片已保存至 {video_path}")

    # 生成分析圖表
    if len(frame_numbers) > 0:
        plt.figure(figsize=(15, 10))

        # 繪製錯誤得分隨時間變化
        plt.subplot(2, 1, 1)
        plt.plot(frame_numbers, error_scores, 'b-', label='預測錯誤得分', fontproperties=chinese_font)

        # 添加真實標籤標記
        for i, label in enumerate(true_labels):
            if label == 1:
                plt.axvline(x=i, color='r', alpha=0.2)

        plt.axhline(y=threshold, color='r', linestyle='--', label=f'閾值 ({threshold})')
        plt.title('錯誤偵測得分隨時間變化', fontproperties=chinese_font)
        plt.xlabel('影格', fontproperties=chinese_font)
        plt.ylabel('錯誤得分', fontproperties=chinese_font)
        plt.legend()
        plt.grid(True)

        # 繪製參數隨時間變化
        plt.subplot(2, 1, 2)
        for param, values in param_values.items():
            if len(values) == len(frame_numbers):
                plt.plot(frame_numbers, values, label=param)

        plt.title('參數隨時間變化', fontproperties=chinese_font)
        plt.xlabel('影格', fontproperties=chinese_font)
        plt.ylabel('參數值', fontproperties=chinese_font)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"分析圖表已保存至 {output_path}")


def plot_error_type_comparison(results, output_path=None):
    """
    繪製不同錯誤類型的檢測效果比較

    Args:
        results: 錯誤類型比較結果字典
        output_path: 輸出路徑
    """
    if not results:
        print("沒有錯誤類型比較結果可繪製")
        return

    # 提取資料
    error_types = list(results.keys())
    counts = [results[e]['count'] for e in error_types]
    accuracies = [results[e]['accuracy'] for e in error_types]
    recalls = [results[e]['recall'] for e in error_types]
    precisions = [results[e]['precision'] for e in error_types]

    # 建立圖表
    plt.figure(figsize=(14, 8))

    # 繪製樣本數量
    plt.subplot(2, 1, 1)
    plt.bar(error_types, counts)
    plt.title('各錯誤類型樣本數量', fontproperties=chinese_font)
    plt.ylabel('樣本數量', fontproperties=chinese_font)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')

    # 繪製效能指標
    plt.subplot(2, 1, 2)
    x = np.arange(len(error_types))
    width = 0.25

    plt.bar(x - width, accuracies, width, label='準確率', fontproperties=chinese_font)
    plt.bar(x, recalls, width, label='召回率', fontproperties=chinese_font)
    plt.bar(x + width, precisions, width, label='精確度', fontproperties=chinese_font)

    plt.title('不同錯誤類型的檢測效果', fontproperties=chinese_font)
    plt.ylabel('效能指標', fontproperties=chinese_font)
    plt.xticks(x, error_types, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"錯誤類型比較已保存至 {output_path}")
    else:
        plt.show()

    plt.close()


def create_confusion_matrix_plot(y_true, y_pred, class_names=['正常', '錯誤'], output_path=None):
    """
    創建混淆矩陣視覺化

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        class_names: 類別名稱
        output_path: 輸出路徑
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)

    # 歸一化
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 設置圖表
    plt.figure(figsize=(10, 8))

    # 繪製熱圖
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('混淆矩陣', fontproperties=chinese_font)
    plt.ylabel('真實標籤', fontproperties=chinese_font)
    plt.xlabel('預測標籤', fontproperties=chinese_font)

    # 計算性能指標
    accuracy = np.trace(cm) / np.sum(cm)
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 添加性能指標
    plt.figtext(0.1, 0.01, f"準確率: {accuracy:.4f}", fontsize=12, fontproperties=chinese_font)
    plt.figtext(0.3, 0.01, f"精確度: {precision:.4f}", fontsize=12, fontproperties=chinese_font)
    plt.figtext(0.5, 0.01, f"召回率: {recall:.4f}", fontsize=12, fontproperties=chinese_font)
    plt.figtext(0.7, 0.01, f"F1得分: {f1:.4f}", fontsize=12, fontproperties=chinese_font)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已保存至 {output_path}")
    else:
        plt.show()

    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }