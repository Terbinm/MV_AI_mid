"""
評估指標實作
用於3D列印錯誤偵測模型的評估
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix, classification_report

from utils.font_utils import chinese_font

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    計算Dice係數
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        smooth: 平滑因子，防止除以零

    Returns:
        dice係數
    """
    # 確保輸入張量為相同的數據類型
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coefficient(y_true, y_pred, smooth=1.0):
    """
    計算IoU係數 (Intersection over Union)
    IoU = |X ∩ Y| / |X ∪ Y|

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        smooth: 平滑因子，防止除以零

    Returns:
        IoU係數
    """
    # 確保輸入張量為相同的數據類型
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coefficient_numpy(y_true, y_pred, threshold=0.5, smooth=1.0):
    """
    使用NumPy計算Dice係數，用於非TensorFlow環境的評估

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        threshold: 二值化閾值
        smooth: 平滑因子，防止除以零

    Returns:
        dice係數
    """
    # 二值化預測結果
    y_pred_bin = (y_pred > threshold).astype(np.float32)

    # 計算Dice係數
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_bin.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_coefficient_numpy(y_true, y_pred, threshold=0.5, smooth=1.0):
    """
    使用NumPy計算IoU係數，用於非TensorFlow環境的評估

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        threshold: 二值化閾值
        smooth: 平滑因子，防止除以零

    Returns:
        IoU係數
    """
    # 二值化預測結果
    y_pred_bin = (y_pred > threshold).astype(np.float32)

    # 計算IoU係數
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_bin.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """
    計算像素準確率

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        threshold: 二值化閾值

    Returns:
        像素準確率
    """
    # 二值化預測結果
    y_pred_bin = (y_pred > threshold).astype(np.float32)

    # 計算準確率
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_bin.flatten()
    correct = np.sum(y_true_f == y_pred_f)
    total = len(y_true_f)
    return correct / total


def evaluate_model(model, test_generator, num_samples, threshold=0.5, save_path=None):
    """
    全面評估模型性能

    Args:
        model: 訓練好的模型
        test_generator: 測試資料產生器
        num_samples: 評估樣本數量
        threshold: 二值化閾值
        save_path: 評估結果保存路徑

    Returns:
        評估指標字典
    """
    print("開始評估模型...")

    # 儲存所有真實標籤和預測結果
    y_true_all = []
    y_pred_all = []

    # 計算批次數量
    steps = min(len(test_generator), num_samples // test_generator.batch_size)

    # 遍歷資料產生器
    for i in range(steps):
        x_batch, y_batch = test_generator[i]
        y_pred_batch = model.predict(x_batch)

        # 儲存結果
        y_true_all.append(y_batch)
        y_pred_all.append(y_pred_batch)

    # 合併所有批次結果
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    # 計算各項指標
    dice = dice_coefficient_numpy(y_true, y_pred, threshold)
    iou = iou_coefficient_numpy(y_true, y_pred, threshold)
    accuracy = pixel_accuracy(y_true, y_pred, threshold)

    # 轉換為圖像級別的二分類問題
    # 如果一張影像中有超過5%的像素被預測為錯誤，則整張影像被視為錯誤
    error_threshold = 0.05
    y_true_img = np.mean(y_true, axis=(1, 2, 3)) > 0.5  # 真實標籤是全1或全0
    y_pred_img = np.mean(y_pred > threshold, axis=(1, 2, 3)) > error_threshold

    # 計算分類報告
    try:
        report = classification_report(y_true_img, y_pred_img,
                                      target_names=['正常', '錯誤'],
                                      output_dict=True,
                                      zero_division=0)
    except Exception as e:
        print(f"無法生成分類報告: {e}")
        report = {}

    # 計算混淆矩陣
    try:
        cm = confusion_matrix(y_true_img, y_pred_img)
    except Exception as e:
        print(f"無法計算混淆矩陣: {e}")
        cm = np.zeros((2, 2))

    # 整理評估結果
    results = {
        'dice_coefficient': float(dice),
        'iou_coefficient': float(iou),
        'pixel_accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    # 打印結果
    print(f"像素準確率: {accuracy:.4f}")
    print(f"Dice係數: {dice:.4f}")
    print(f"IoU係數: {iou:.4f}")
    print("\n圖像級別的分類性能:")
    print(f"準確率: {(cm[0,0] + cm[1,1]) / cm.sum():.4f}")
    print(f"敏感度 (召回率): {cm[1,1] / (cm[1,0] + cm[1,1]):.4f}")
    print(f"特異度: {cm[0,0] / (cm[0,0] + cm[0,1]):.4f}")
    print(f"精確度: {cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0:.4f}")

    # 保存結果
    if save_path:
        import json
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

        # 繪製並保存混淆矩陣
        plot_path = os.path.join(os.path.dirname(save_path), 'confusion_matrix.png')
        plot_confusion_matrix(cm, ['正常', '錯誤'], plot_path)

        print(f"評估結果已保存至 {save_path}")

    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    繪製混淆矩陣

    Args:
        cm: 混淆矩陣
        class_names: 類別名稱列表
        save_path: 保存路徑
    """
    fig, ax = plt.figure(figsize=(8, 6)), plt.subplot(111)

    # 繪製混淆矩陣
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    # 設置軸位置和標籤
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontproperties=chinese_font)
    ax.set_yticklabels(class_names, fontproperties=chinese_font)

    # 添加數值標籤
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center')

    # 設置標題和軸標籤
    plt.title('混淆矩陣', fontproperties=chinese_font)
    plt.xlabel('預測標籤', fontproperties=chinese_font)
    plt.ylabel('真實標籤', fontproperties=chinese_font)

    # 保存圖像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已保存至 {save_path}")

    plt.close()



def visualize_predictions(model, test_generator, num_samples=10, threshold=0.5, save_dir=None):
    """
    視覺化模型預測結果

    Args:
        model: 訓練好的模型
        test_generator: 測試資料產生器
        num_samples: 視覺化樣本數量
        threshold: 二值化閾值
        save_dir: 結果保存目錄
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 隨機選擇批次
    batch_idx = np.random.randint(0, len(test_generator))
    x_batch, y_batch = test_generator[batch_idx]
    y_pred_batch = model.predict(x_batch)

    # 確保樣本數量不超過批次大小
    num_samples = min(num_samples, x_batch.shape[0])

    for i in range(num_samples):
        # 獲取原始影像、真實遮罩和預測遮罩
        img = x_batch[i]
        true_mask = y_batch[i].squeeze()
        pred_mask = y_pred_batch[i].squeeze()

        # 反正規化影像 (如果已正規化)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # 二值化預測遮罩
        pred_mask_bin = (pred_mask > threshold).astype(np.uint8)

        # 創建覆蓋效果
        overlay_pred = img.copy()
        overlay_true = img.copy()

        # 使用綠色覆蓋預測的錯誤區域
        overlay_pred[pred_mask_bin > 0, 0] = overlay_pred[pred_mask_bin > 0, 0] * 0.5
        overlay_pred[pred_mask_bin > 0, 1] = overlay_pred[pred_mask_bin > 0, 1] * 0.5 + 128
        overlay_pred[pred_mask_bin > 0, 2] = overlay_pred[pred_mask_bin > 0, 2] * 0.5

        # 使用紅色覆蓋真實的錯誤區域
        overlay_true[true_mask > 0, 0] = overlay_true[true_mask > 0, 0] * 0.5 + 128
        overlay_true[true_mask > 0, 1] = overlay_true[true_mask > 0, 1] * 0.5
        overlay_true[true_mask > 0, 2] = overlay_true[true_mask > 0, 2] * 0.5

        # 計算該樣本的Dice和IoU
        dice = dice_coefficient_numpy(true_mask, pred_mask, threshold)
        iou = iou_coefficient_numpy(true_mask, pred_mask, threshold)

        # 創建四個子圖 (原始影像、真實遮罩、預測遮罩、疊加效果)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(img)
        axes[0, 0].set_title('原始影像', fontproperties=chinese_font)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(true_mask, cmap='gray')
        axes[0, 1].set_title('真實遮罩' + (' (有錯誤)' if true_mask.max() > 0 else ' (無錯誤)'), fontproperties=chinese_font)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(pred_mask, cmap='gray')
        axes[1, 0].set_title(f'預測遮罩 (Dice={dice:.4f}, IoU={iou:.4f})', fontproperties=chinese_font)
        axes[1, 0].axis('off')

        # 疊加效果：綠色表示預測錯誤，紅色表示真實錯誤
        combined = img.copy()
        combined[true_mask > 0, 0] = combined[true_mask > 0, 0] * 0.5 + 128  # 紅色半透明
        combined[true_mask > 0, 1] = combined[true_mask > 0, 1] * 0.5
        combined[true_mask > 0, 2] = combined[true_mask > 0, 2] * 0.5

        combined[pred_mask_bin > 0, 0] = combined[pred_mask_bin > 0, 0] * 0.5
        combined[pred_mask_bin > 0, 1] = combined[pred_mask_bin > 0, 1] * 0.5 + 128  # 綠色半透明
        combined[pred_mask_bin > 0, 2] = combined[pred_mask_bin > 0, 2] * 0.5

        axes[1, 1].imshow(combined)
        axes[1, 1].set_title('疊加效果 (紅:真實錯誤, 綠:預測錯誤)', fontproperties=chinese_font)
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 保存結果
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def learning_curves(history, save_path=None):
    """
    繪製學習曲線

    Args:
        history: 訓練歷史記錄
        save_path: 保存路徑
    """
    # 創建包含兩個子圖的圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 繪製損失曲線
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 繪製指標曲線 (Dice或IoU)
    for metric in history.history:
        if metric not in ['loss', 'val_loss']:
            if 'val_' not in metric:  # 只處理訓練指標，驗證指標在下一次迴圈中處理
                train_metric = metric
                val_metric = 'val_' + metric

                ax2.plot(history.history[train_metric], label=f'Training {train_metric}')
                ax2.plot(history.history[val_metric], label=f'Validation {val_metric}')

    ax2.set_title('Metric Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # 保存圖表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"學習曲線已保存至 {save_path}")

    plt.close()


def find_threshold(y_true, y_pred, threshold_range=np.arange(0.1, 0.9, 0.05)):
    """
    尋找最佳二值化閾值

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        threshold_range: 待測試的閾值範圍

    Returns:
        最佳閾值和對應的Dice係數
    """
    best_dice = 0
    best_threshold = 0.5

    for threshold in threshold_range:
        dice = dice_coefficient_numpy(y_true, y_pred, threshold)
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold

    return best_threshold, best_dice


def compare_error_types(model, test_generator, num_samples=100, threshold=0.5, save_path=None):
    """
    比較不同類型的錯誤檢測效果

    Args:
        model: 訓練好的模型
        test_generator: 測試資料產生器
        num_samples: 評估樣本數量
        threshold: 二值化閾值
        save_path: 保存路徑

    Returns:
        不同錯誤類型的評估結果
    """
    # 評估樣本數量不應超過資料集大小
    steps = min(len(test_generator), num_samples // test_generator.batch_size)

    # 儲存結果
    all_predictions = []
    all_true_labels = []
    error_types = []

    # 遍歷批次
    for i in range(steps):
        x_batch, y_batch = test_generator[i]
        y_pred_batch = model.predict(x_batch)

        # 取得該批次的索引
        batch_indices = test_generator.indices[i*test_generator.batch_size:(i+1)*test_generator.batch_size]

        # 獲取對應的CSV資料
        if hasattr(test_generator, 'df'):
            df_batch = test_generator.df.loc[batch_indices]

            # 提取每個樣本的錯誤類型
            for idx, row in df_batch.iterrows():
                error_type = []
                if row['flow_rate_class'] != 1:
                    error_type.append('flow_rate')
                if row['feed_rate_class'] != 1:
                    error_type.append('feed_rate')
                if row['z_offset_class'] != 1:
                    error_type.append('z_offset')
                if row['hotend_class'] != 1:
                    error_type.append('hotend')

                error_types.append('_'.join(error_type) if error_type else 'normal')

        # 儲存預測結果
        for j in range(len(x_batch)):
            true_mask = y_batch[j].squeeze()
            pred_mask = y_pred_batch[j].squeeze()

            # 轉換為二分類
            true_label = 1 if np.mean(true_mask) > 0.5 else 0
            pred_label = 1 if np.mean(pred_mask > threshold) > 0.05 else 0

            all_true_labels.append(true_label)
            all_predictions.append(pred_label)

    # 如果沒有錯誤類型信息，則結束
    if not error_types:
        print("無法獲取錯誤類型信息")
        return None

    # 將結果轉換為NumPy陣列
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # 計算每種錯誤類型的檢測效果
    error_results = {}
    unique_errors = np.unique(error_types)

    for error in unique_errors:
        mask = np.array(error_types) == error
        if mask.sum() == 0:
            continue

        y_true_subset = all_true_labels[mask]
        y_pred_subset = all_predictions[mask]

        # 計算該錯誤類型的檢測效果
        accuracy = np.mean(y_true_subset == y_pred_subset)
        if y_true_subset.sum() > 0:  # 避免除以零
            recall = np.sum((y_true_subset == 1) & (y_pred_subset == 1)) / y_true_subset.sum()
        else:
            recall = np.nan

        if y_pred_subset.sum() > 0:  # 避免除以零
            precision = np.sum((y_true_subset == 1) & (y_pred_subset == 1)) / y_pred_subset.sum()
        else:
            precision = np.nan

        error_results[error] = {
            'count': mask.sum(),
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        }

    # 打印結果
    print("\n不同錯誤類型的檢測效果:")
    for error, metrics in error_results.items():
        print(f"{error}: 樣本數={metrics['count']}, 準確率={metrics['accuracy']:.4f}, "
              f"召回率={metrics['recall']:.4f}, 精確度={metrics['precision']:.4f}")

    # 繪製結果
    if save_path:
        # 繪製柱狀圖
        error_types = list(error_results.keys())
        accuracies = [error_results[e]['accuracy'] for e in error_types]
        recalls = [error_results[e]['recall'] for e in error_types]
        precisions = [error_results[e]['precision'] for e in error_types]

        x = np.arange(len(error_types))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, accuracies, width, label='Accuracy')
        ax.bar(x, recalls, width, label='Recall')
        ax.bar(x + width, precisions, width, label='Precision')

        ax.set_ylabel('Score')
        ax.set_title('不同錯誤類型的檢測效果', fontproperties=chinese_font)
        ax.set_xticks(x)
        ax.set_xticklabels(error_types, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"錯誤類型比較結果已保存至 {save_path}")
        plt.close()

    return error_results