"""
評估腳本
用於評估3D列印錯誤偵測模型性能
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from config.config import Config
from models.unet import dice_coef as dice_coefficient, iou_coef as iou_coefficient
from models.unet import  dice_loss, combined_loss
from utils.dataset import load_dataset_from_indices
from utils.metrics import evaluate_model, visualize_predictions, compare_error_types
from utils.visualization import (
    create_confusion_matrix_plot, plot_error_type_comparison
)
from utils.font_utils import chinese_font


def load_trained_model(model_path):
    """
    載入訓練好的模型

    Args:
        model_path: 模型檔案路徑

    Returns:
        model: 載入的模型
    """
    print(f"載入模型: {model_path}")

    # 載入自定義指標和損失函數
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'iou_coefficient': iou_coefficient,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss
    }

    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("模型載入成功")
        return model
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        sys.exit(1)


def evaluate_on_validation_set(model, config, output_dir):
    """
    在驗證集上評估模型

    Args:
        model: 訓練好的模型
        config: 配置對象
        output_dir: 輸出目錄
    """
    print("在驗證集上評估模型...")

    # 載入驗證資料
    val_indices_file = os.path.join(config.DATA_DIR, "processed", "val_indices.csv")
    val_dir = os.path.join(config.DATA_DIR, "processed", "val")

    if not os.path.exists(val_indices_file) or not os.path.exists(val_dir):
        print(f"錯誤: 找不到驗證集資料，請確認已執行資料預處理")
        return

    val_generator = load_dataset_from_indices(
        processed_dir=val_dir,
        indices_file=val_indices_file,
        batch_size=config.BATCH_SIZE,
        is_training=False,
        input_size=config.INPUT_SIZE[:2]
    )

    # 評估模型
    evaluation_results = evaluate_model(
        model=model,
        test_generator=val_generator,
        num_samples=len(val_generator) * config.BATCH_SIZE,
        threshold=config.THRESHOLD,
        save_path=os.path.join(output_dir, "validation_results.json")
    )

    # 視覺化預測結果
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    visualize_predictions(
        model=model,
        test_generator=val_generator,
        num_samples=10,
        threshold=config.THRESHOLD,
        save_dir=os.path.join(output_dir, "visualizations")
    )

    # 比較不同錯誤類型的檢測效果
    if hasattr(val_generator, 'df'):
        error_results = compare_error_types(
            model=model,
            test_generator=val_generator,
            num_samples=len(val_generator) * config.BATCH_SIZE,
            threshold=config.THRESHOLD,
            save_path=os.path.join(output_dir, "error_type_comparison.png")
        )

        if error_results:
            plot_error_type_comparison(
                error_results,
                output_path=os.path.join(output_dir, "error_type_plot.png")
            )

    # 繪製ROC曲線
    try:
        evaluate_thresholds(model, val_generator, output_dir)
    except Exception as e:
        print(f"繪製ROC曲線時發生錯誤: {e}")

    return evaluation_results


def evaluate_with_custom_csv(model, csv_path, image_dir, config, output_dir):
    """
    使用自定義CSV檔案評估模型

    Args:
        model: 訓練好的模型
        csv_path: CSV檔案路徑
        image_dir: 影像目錄
        config: 配置對象
        output_dir: 輸出目錄
    """
    print(f"使用自定義CSV評估模型: {csv_path}")

    # 讀取CSV檔案
    df = pd.read_csv(csv_path)

    if 'img_path' not in df.columns:
        print(f"錯誤: CSV檔案中找不到'img_path'欄位")
        return

    print(f"CSV檔案中有 {len(df)} 筆資料")

    # 創建錯誤標籤(如果不存在)
    if 'error_label' not in df.columns and all(
            col in df.columns for col in ['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']):
        df['error_label'] = ((df['flow_rate_class'] != 1) |
                             (df['feed_rate_class'] != 1) |
                             (df['z_offset_class'] != 1) |
                             (df['hotend_class'] != 1)).astype(int)

    # 檢查是否有錯誤標籤
    if 'error_label' not in df.columns:
        print(f"錯誤: CSV檔案中找不到'error_label'欄位或無法從參數類別建立標籤")
        return

    # 預測並評估
    y_true = []
    y_pred = []
    y_scores = []

    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, row['img_path'])

        if not os.path.exists(img_path):
            print(f"警告: 找不到影像 {img_path}")
            continue

        # 處理影像
        if 'nozzle_tip_x' in row and 'nozzle_tip_y' in row:
            # 讀取影像
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 無法讀取影像 {img_path}")
                continue

            # 裁剪噴嘴區域
            nozzle_x = int(row['nozzle_tip_x'])
            nozzle_y = int(row['nozzle_tip_y'])

            crop_size = config.CROP_SIZE
            x1 = max(0, nozzle_x - crop_size // 2)
            y1 = max(0, nozzle_y - crop_size // 2)
            x2 = min(img.shape[1], x1 + crop_size)
            y2 = min(img.shape[0], y1 + crop_size)

            # 調整裁剪區域
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)

            cropped = img[y1:y2, x1:x2]

            # 調整大小並轉換顏色空間
            img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
        else:
            # 直接讀取並調整大小
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 無法讀取影像 {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))

        # 正規化
        img_norm = img_resized.astype(np.float32) / 255.0

        # 預測
        pred = model.predict(np.expand_dims(img_norm, axis=0))[0]

        # 計算錯誤得分
        error_score = float(np.mean(pred.squeeze()))
        error_pred = 1 if error_score > config.THRESHOLD else 0

        # 記錄結果
        y_true.append(row['error_label'])
        y_pred.append(error_pred)
        y_scores.append(error_score)

    # 計算評估指標
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n評估指標:")
    print(f"準確率: {acc:.4f}")
    print(f"精確度: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1分數: {f1:.4f}")

    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    print("\n混淆矩陣:")
    print(f"True Negative: {cm[0, 0]}, False Positive: {cm[0, 1]}")
    print(f"False Negative: {cm[1, 0]}, True Positive: {cm[1, 1]}")

    # 分類報告
    print("\n分類報告:")
    print(classification_report(y_true, y_pred, target_names=['正常', '錯誤']))

    # 將結果保存
    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'threshold': config.THRESHOLD
    }

    import json
    with open(os.path.join(output_dir, "custom_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # 繪製混淆矩陣
    create_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['正常', '錯誤'],
        output_path=os.path.join(output_dir, "custom_confusion_matrix.png")
    )

    # 繪製ROC曲線
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲線 (面積 = {roc_auc:.4f})', fontproperties=chinese_font)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率 (False Positive Rate)', fontproperties=chinese_font)
    plt.ylabel('真陽性率 (True Positive Rate)', fontproperties=chinese_font)
    plt.title('接收者操作特徵曲線 (ROC)', fontproperties=chinese_font)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "custom_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return results


def evaluate_thresholds(model, val_generator, output_dir):
    """
    評估不同閾值下的模型性能

    Args:
        model: 訓練好的模型
        val_generator: 驗證資料產生器
        output_dir: 輸出目錄
    """
    print("評估不同閾值下的模型性能...")

    # 收集預測和真實標籤
    y_true = []
    y_scores = []

    for i in range(min(10, len(val_generator))):  # 限制數量以加快評估速度
        x_batch, y_batch = val_generator[i]
        y_pred_batch = model.predict(x_batch)

        for j in range(len(y_batch)):
            y_true.append(1 if np.mean(y_batch[j]) > 0.5 else 0)
            y_scores.append(float(np.mean(y_pred_batch[j])))

    # 計算ROC曲線
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 繪製ROC曲線
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲線 (面積 = {roc_auc:.4f})', fontproperties=chinese_font)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('偽陽性率 (False Positive Rate)', fontproperties=chinese_font)
    plt.ylabel('真陽性率 (True Positive Rate)', fontproperties=chinese_font)
    plt.title('接收者操作特徵曲線 (ROC)', fontproperties=chinese_font)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 計算不同閾值下的性能指標
    thresholds_to_evaluate = np.arange(0.1, 0.9, 0.05)
    results = []

    for threshold in thresholds_to_evaluate:
        y_pred = [1 if score > threshold else 0 for score in y_scores]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

    # 將結果轉換為DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "threshold_evaluation.csv"), index=False)

    # 繪製不同閾值下的性能變化
    plt.figure(figsize=(12, 8))
    plt.plot(df_results['threshold'], df_results['accuracy'], 'o-', label='準確率', fontproperties=chinese_font)
    plt.plot(df_results['threshold'], df_results['precision'], 'o-', label='精確度', fontproperties=chinese_font)
    plt.plot(df_results['threshold'], df_results['recall'], 'o-', label='召回率', fontproperties=chinese_font)
    plt.plot(df_results['threshold'], df_results['f1_score'], 'o-', label='F1分數', fontproperties=chinese_font)
    plt.xlabel('閾值', fontproperties=chinese_font)
    plt.ylabel('指標值', fontproperties=chinese_font)
    plt.title('不同閾值下的模型性能', fontproperties=chinese_font)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "threshold_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 找出最佳F1分數的閾值
    best_f1_idx = df_results['f1_score'].idxmax()
    best_threshold = df_results.loc[best_f1_idx, 'threshold']
    best_f1 = df_results.loc[best_f1_idx, 'f1_score']

    print(f"最佳F1分數 ({best_f1:.4f}) 對應的閾值為: {best_threshold}")

    # 將最佳閾值保存
    with open(os.path.join(output_dir, "best_threshold.txt"), 'w') as f:
        f.write(f"Best threshold: {best_threshold}\n")
        f.write(f"F1 score: {best_f1}\n")
        f.write(f"Accuracy: {df_results.loc[best_f1_idx, 'accuracy']}\n")
        f.write(f"Precision: {df_results.loc[best_f1_idx, 'precision']}\n")
        f.write(f"Recall: {df_results.loc[best_f1_idx, 'recall']}\n")

    return best_threshold, df_results


def main():
    """
    主函數
    """
    # 載入配置
    config = Config()

    parser = argparse.ArgumentParser(description='3D列印錯誤偵測模型評估腳本')
    parser.add_argument('--model', type=str, required=True, help='訓練好的模型路徑')
    parser.add_argument('--mode', type=str, choices=['validation', 'custom'], default='validation',
                        help='評估模式: validation(驗證集), custom(自定義資料集)')
    parser.add_argument('--csv', type=str, help='自定義評估CSV檔案路徑 (僅用於custom模式)')
    parser.add_argument('--image_dir', type=str, help='影像根目錄 (僅用於custom模式)')
    parser.add_argument('--output', type=str, default='./results/evaluation', help='輸出目錄')
    parser.add_argument('--threshold', type=float, default=None, help='二值化閾值')

    args = parser.parse_args()

    # 更新配置閾值
    if args.threshold is not None:
        config.THRESHOLD = args.threshold

    # 設置GPU記憶體增長 - 移至模型載入前
    if config.GPU_MEMORY_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    print(f"已啟用GPU記憶體增長")
                except Exception as e:
                    print(f"設置GPU記憶體增長時發生錯誤: {e}")

    # 載入模型
    model = load_trained_model(args.model)

    # 建立輸出目錄
    os.makedirs(args.output, exist_ok=True)

    # 根據模式進行評估
    if args.mode == 'validation':
        evaluate_on_validation_set(model, config, args.output)

    elif args.mode == 'custom':
        if not args.csv or not args.image_dir:
            print("錯誤: custom模式需要同時指定 --csv 和 --image_dir 參數")
            return

        evaluate_with_custom_csv(model, args.csv, args.image_dir, config, args.output)

    print(f"評估完成，結果已保存至: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"評估過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()