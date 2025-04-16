"""
預測腳本 - 分類器版本
用於使用訓練好的模型進行3D列印錯誤分類
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
import pandas as pd
from tensorflow.keras.models import load_model

from config.config import Config
from models.unet import get_unet_with_custom_params
from utils.visualization import create_error_visualization_video
from utils.font_utils import chinese_font


def load_trained_classifier(model_path):
    """
    載入訓練好的分類器模型

    Args:
        model_path: 模型檔案路徑

    Returns:
        model: 載入的模型
    """
    print(f"載入分類器模型: {model_path}")

    # 載入自定義指標
    custom_objects = {
        'accuracy': tf.keras.metrics.binary_accuracy,
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
        'auc': tf.keras.metrics.AUC()
    }

    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("模型載入成功")
        return model
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        sys.exit(1)


def predict_single_image(model, image_path, config):
    """
    對單張影像進行分類預測

    Args:
        model: 訓練好的模型
        image_path: 影像路徑
        config: 配置對象

    Returns:
        原始影像和預測結果（錯誤機率）
    """
    # 讀取影像
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取影像: {image_path}")
        return None, None

    # 轉換顏色空間
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 調整大小
    img_resized = cv2.resize(img, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))

    # 正規化
    img_norm = img_resized.astype(np.float32) / 255.0

    # 預測
    pred = model.predict(np.expand_dims(img_norm, axis=0))[0][0]  # 單一值

    return img_resized, pred


def predict_from_directory(model, input_dir, output_dir, config):
    """
    對目錄中的所有影像進行分類預測

    Args:
        model: 訓練好的模型
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        config: 配置對象
    """
    print(f"從目錄進行批次分類預測: {input_dir}")

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 獲取所有影像檔案
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg"))

    if len(image_files) == 0:
        print(f"目錄中沒有發現影像文件: {input_dir}")
        return

    print(f"發現 {len(image_files)} 個影像檔案")

    # 逐一處理影像
    results = []
    for image_path in tqdm(image_files, desc="預測影像"):
        img_resized, error_score = predict_single_image(model, image_path, config)

        if img_resized is None:
            continue

        # 獲取檔案名稱（不含副檔名）
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # 二值化判斷
        error_status = "錯誤" if error_score > config.THRESHOLD else "正常"

        # 創建可視化
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized)
        plt.title('原始影像', fontproperties=chinese_font)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        # 顯示預測結果
        plt.text(0.5, 0.5, f'預測: {error_status}\n準確度: {error_score:.4f}',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, fontproperties=chinese_font,
                 color='red' if error_score > config.THRESHOLD else 'green')
        plt.axis('off')

        # 保存可視化
        viz_path = os.path.join(output_dir, f"{file_name}_prediction.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 記錄結果
        results.append({
            "file_name": file_name,
            "image_path": image_path,
            "visualization_path": viz_path,
            "error_score": float(error_score),
            "error_status": error_status
        })

    # 保存結果摘要
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)

    # 繪製結果分布圖
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['error_score'], bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=config.THRESHOLD, color='r', linestyle='--', label=f'閾值 ({config.THRESHOLD})')
    plt.title('錯誤得分分布', fontproperties=chinese_font)
    plt.xlabel('錯誤得分', fontproperties=chinese_font)
    plt.ylabel('頻率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_score_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"預測完成，結果已保存至: {output_dir}")
    print(f"正常影像: {sum(results_df['error_status'] == '正常')} 張")
    print(f"錯誤影像: {sum(results_df['error_status'] == '錯誤')} 張")


def predict_from_csv(model, csv_path, image_dir, output_dir, config, visualization=True):
    """
    根據CSV檔案對影像進行分類預測

    Args:
        model: 訓練好的模型
        csv_path: CSV檔案路徑
        image_dir: 影像目錄
        output_dir: 輸出目錄
        config: 配置對象
        visualization: 是否生成視覺化影片
    """
    print(f"從CSV檔案進行分類預測: {csv_path}")

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

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

    # 初始化結果列表
    predictions = []

    # 逐一處理影像
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="預測影像"):
        img_path = os.path.join(image_dir, row['img_path'])

        if not os.path.exists(img_path):
            print(f"警告: 找不到影像 {img_path}")
            continue

        # 處理噴嘴區域影像
        if 'nozzle_tip_x' in row and 'nozzle_tip_y' in row:
            # 讀取完整影像
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
            img_resized, _ = predict_single_image(model, img_path, config)
            if img_resized is None:
                continue

        # 正規化
        img_norm = img_resized.astype(np.float32) / 255.0

        # 預測
        error_score = float(model.predict(np.expand_dims(img_norm, axis=0))[0][0])
        error_status = "錯誤" if error_score > config.THRESHOLD else "正常"

        # 保存結果
        result = {
            'index': idx,
            'img_path': row['img_path'],
            'error_score': error_score,
            'error_status': error_status
        }

        # 如果有真實標籤，計算準確性
        if 'error_label' in row:
            true_label = row['error_label']
            true_status = "錯誤" if true_label == 1 else "正常"
            correct = (error_status == true_status)

            result['true_label'] = true_label
            result['true_status'] = true_status
            result['correct'] = correct

        predictions.append(result)

        # 保存預測結果可視化
        if idx < 100:  # 只保存前100張的視覺化結果，避免佔用太多空間
            # 創建並保存圖表
            plt.figure(figsize=(10, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(img_resized)
            plt.title('原始影像', fontproperties=chinese_font)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            # 創建預測結果文字
            info_text = [
                f"預測: {error_status} (得分: {error_score:.3f})"
            ]

            # 如果有真實標籤，添加相關信息
            if 'error_label' in row:
                info_text.append(f"真實: {true_status}")
                info_text.append(f"正確: {'是' if correct else '否'}")

            # 添加參數信息
            for param in ['flow_rate', 'feed_rate', 'z_offset', 'hotend']:
                if param in row and f"{param}_class" in row:
                    info_text.append(f"{param}: {row[param]} (類別: {row[f'{param}_class']})")

            plt.text(0.5, 0.5, '\n'.join(info_text),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=11, fontproperties=chinese_font,
                     color='red' if error_score > config.THRESHOLD else 'green')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "predictions", f"pred_{idx}.png"))
            plt.close()

    # 保存預測結果
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # 計算統計資訊
    print(f"\n預測結果統計:")
    print(f"總樣本數: {len(predictions_df)}")
    print(
        f"預測為正常: {sum(predictions_df['error_status'] == '正常')} ({sum(predictions_df['error_status'] == '正常') / len(predictions_df) * 100:.2f}%)")
    print(
        f"預測為錯誤: {sum(predictions_df['error_status'] == '錯誤')} ({sum(predictions_df['error_status'] == '錯誤') / len(predictions_df) * 100:.2f}%)")

    if 'true_label' in predictions_df.columns:
        correct_count = sum(predictions_df['correct'])
        accuracy = correct_count / len(predictions_df)

        print(f"準確率: {accuracy:.4f} ({correct_count}/{len(predictions_df)})")

        # 計算混淆矩陣
        from sklearn.metrics import confusion_matrix, classification_report

        y_true = predictions_df['true_label'].values
        y_pred = (predictions_df['error_status'] == '錯誤').astype(int).values

        cm = confusion_matrix(y_true, y_pred)
        print("\n混淆矩陣:")
        print(f"True Negative: {cm[0, 0]}, False Positive: {cm[0, 1]}")
        print(f"False Negative: {cm[1, 0]}, True Positive: {cm[1, 1]}")

        print("\n分類報告:")
        print(classification_report(y_true, y_pred, target_names=['正常', '錯誤']))

        # 將混淆矩陣視覺化
        from utils.visualization import create_confusion_matrix_plot
        create_confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            class_names=['正常', '錯誤'],
            output_path=os.path.join(output_dir, "confusion_matrix.png")
        )

    # 繪製錯誤分數分布
    plt.figure(figsize=(10, 6))
    plt.hist(predictions_df['error_score'], bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=config.THRESHOLD, color='r', linestyle='--', label=f'閾值 ({config.THRESHOLD})')
    plt.title('錯誤得分分布', fontproperties=chinese_font)
    plt.xlabel('錯誤得分', fontproperties=chinese_font)
    plt.ylabel('頻率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "error_score_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 創建視覺化影片
    if visualization and len(df) > 10:
        print("\n創建視覺化影片...")
        try:
            create_error_visualization_video(
                model=model,
                video_path=os.path.join(output_dir, "error_detection_video.mp4"),
                csv_path=csv_path,
                image_dir=image_dir,
                output_path=os.path.join(output_dir, "error_detection_summary.png"),
                fps=5,
                threshold=config.THRESHOLD,
                is_classifier=True  # 標記為分類器模型
            )
        except Exception as e:
            print(f"創建視覺化影片時發生錯誤: {e}")

    print(f"\n預測完成，結果已保存至: {output_dir}")


def main():
    """
    主函數
    """
    # 載入配置
    config = Config()

    parser = argparse.ArgumentParser(description='3D列印錯誤分類預測腳本')
    parser.add_argument('--model', type=str, required=True, help='訓練好的分類器模型路徑')
    parser.add_argument('--mode', type=str, choices=['csv', 'directory', 'validation'], default='validation',
                        help='預測模式: csv(依CSV檔案), directory(影像目錄), validation(驗證集)')
    parser.add_argument('--input', type=str, help='輸入路徑 (CSV檔案或影像目錄)')
    parser.add_argument('--image_dir', type=str, help='影像根目錄 (用於CSV模式)')
    parser.add_argument('--output', type=str, default='./results/predictions', help='輸出目錄')
    parser.add_argument('--threshold', type=float, default=None, help='二值化閾值')
    parser.add_argument('--no_video', action='store_true', help='禁用視覺化影片生成')

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
    model = load_trained_classifier(args.model)

    # 根據模式進行預測
    if args.mode == 'validation':
        print("使用驗證集進行預測...")

        # 載入驗證資料
        from utils.dataset import load_dataset_for_classification

        val_indices_file = os.path.join(config.DATA_DIR, "processed", "val_indices.csv")
        val_dir = os.path.join(config.DATA_DIR, "processed", "val")

        if not os.path.exists(val_indices_file) or not os.path.exists(val_dir):
            print(f"錯誤: 找不到驗證集資料，請確認已執行資料預處理")
            return

        val_generator = load_dataset_for_classification(
            processed_dir=val_dir,
            indices_file=val_indices_file,
            batch_size=config.BATCH_SIZE,
            is_training=False,
            input_size=config.INPUT_SIZE[:2]
        )

        # 在驗證集上評估模型
        y_true = []
        y_pred = []

        for i in range(len(val_generator)):
            x_batch, y_batch = val_generator[i]
            predictions = model.predict(x_batch)

            # 二值化預測結果
            binary_preds = (predictions > config.THRESHOLD).astype(int).flatten()

            y_true.extend(y_batch)
            y_pred.extend(binary_preds)

        # 計算評估指標
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
            classification_report

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"準確率: {accuracy:.4f}")
        print(f"精確度: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分數: {f1:.4f}")

        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        print("\n混淆矩陣:")
        print(f"True Negative: {cm[0, 0]}, False Positive: {cm[0, 1]}")
        print(f"False Negative: {cm[1, 0]}, True Positive: {cm[1, 1]}")

        # 分類報告
        report = classification_report(y_true, y_pred, target_names=['正常', '錯誤'])
        print("\n分類報告:")
        print(report)

        # 保存結果
        os.makedirs(args.output, exist_ok=True)

        # 儲存混淆矩陣圖
        from utils.visualization import create_confusion_matrix_plot
        create_confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            class_names=['正常', '錯誤'],
            output_path=os.path.join(args.output, "confusion_matrix.png")
        )

        # 儲存評估結果
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'threshold': config.THRESHOLD
        }

        import json
        with open(os.path.join(args.output, "evaluation_results.json"), 'w') as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(args.output, "classification_report.txt"), 'w') as f:
            f.write(report)

    elif args.mode == 'directory':
        if not args.input:
            print("錯誤: 目錄模式需要指定 --input 參數")
            return

        predict_from_directory(model, args.input, args.output, config)

    elif args.mode == 'csv':
        if not args.input or not args.image_dir:
            print("錯誤: CSV模式需要同時指定 --input 和 --image_dir 參數")
            return

        predict_from_csv(
            model=model,
            csv_path=args.input,
            image_dir=args.image_dir,
            output_dir=args.output,
            config=config,
            visualization=not args.no_video
        )

    print("預測程序執行完畢")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"預測過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()