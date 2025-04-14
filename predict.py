"""
預測腳本
用於使用訓練好的模型進行3D列印錯誤偵測
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
from models.unet import dice_coef as dice_coefficient, iou_coef as iou_coefficient
from utils.dataset import load_dataset_from_indices
from utils.metrics import evaluate_model, visualize_predictions
from utils.visualization import create_error_visualization_video
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

    # 載入自定義指標
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'iou_coefficient': iou_coefficient
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
    對單張影像進行預測

    Args:
        model: 訓練好的模型
        image_path: 影像路徑
        config: 配置對象

    Returns:
        原始影像、預測遮罩和二值化遮罩
    """
    # 讀取影像
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取影像: {image_path}")
        return None, None, None

    # 轉換顏色空間
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 調整大小
    img_resized = cv2.resize(img, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))

    # 正規化
    img_norm = img_resized.astype(np.float32) / 255.0

    # 預測
    pred = model.predict(np.expand_dims(img_norm, axis=0))[0]

    # 二值化
    pred_binary = (pred > config.THRESHOLD).astype(np.uint8)

    return img_resized, pred, pred_binary


def predict_from_directory(model, input_dir, output_dir, config):
    """
    對目錄中的所有影像進行預測

    Args:
        model: 訓練好的模型
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        config: 配置對象
    """
    print(f"從目錄進行批次預測: {input_dir}")

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)

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
        img_resized, pred, pred_binary = predict_single_image(model, image_path, config)

        if img_resized is None:
            continue

        # 獲取檔案名稱（不含副檔名）
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # 保存預測遮罩
        mask_path = os.path.join(output_dir, "masks", f"{file_name}_mask.png")
        cv2.imwrite(mask_path, (pred.squeeze() * 255).astype(np.uint8))

        # 創建並保存疊加效果
        overlay = img_resized.copy()
        pred_bin = pred_binary.squeeze()

        # 綠色半透明覆蓋錯誤區域
        overlay[pred_bin > 0, 0] = overlay[pred_bin > 0, 0] * 0.5
        overlay[pred_bin > 0, 1] = overlay[pred_bin > 0, 1] * 0.5 + 128
        overlay[pred_bin > 0, 2] = overlay[pred_bin > 0, 2] * 0.5

        overlay_path = os.path.join(output_dir, "overlay", f"{file_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 計算錯誤得分(覆蓋率)
        error_score = float(np.mean(pred.squeeze()))
        error_status = "錯誤" if error_score > config.THRESHOLD else "正常"

        # 記錄結果
        results.append({
            "file_name": file_name,
            "image_path": image_path,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "error_score": error_score,
            "error_status": error_status
        })

    # 保存結果摘要
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)

    print(f"預測完成，結果已保存至: {output_dir}")
    print(f"正常影像: {sum(results_df['error_status'] == '正常')} 張")
    print(f"錯誤影像: {sum(results_df['error_status'] == '錯誤')} 張")


def predict_from_csv(model, csv_path, image_dir, output_dir, config, visualization=True):
    """
    根據CSV檔案對影像進行預測

    Args:
        model: 訓練好的模型
        csv_path: CSV檔案路徑
        image_dir: 影像目錄
        output_dir: 輸出目錄
        config: 配置對象
        visualization: 是否生成視覺化影片
    """
    print(f"從CSV檔案進行預測: {csv_path}")

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
            img_resized, _, _ = predict_single_image(model, img_path, config)
            if img_resized is None:
                continue

        # 正規化
        img_norm = img_resized.astype(np.float32) / 255.0

        # 預測
        pred = model.predict(np.expand_dims(img_norm, axis=0))[0]

        # 計算錯誤得分
        error_score = float(np.mean(pred.squeeze()))
        error_status = "錯誤" if error_score > config.THRESHOLD else "正常"

        # 二值化
        pred_binary = (pred > config.THRESHOLD).astype(np.uint8)

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

        # 保存預測結果
        if idx < 100:  # 只保存前100張的視覺化結果，避免佔用太多空間
            # 創建疊加效果
            overlay = img_resized.copy()
            pred_bin = pred_binary.squeeze()

            # 綠色半透明覆蓋錯誤區域
            overlay[pred_bin > 0, 0] = overlay[pred_bin > 0, 0] * 0.5
            overlay[pred_bin > 0, 1] = overlay[pred_bin > 0, 1] * 0.5 + 128
            overlay[pred_bin > 0, 2] = overlay[pred_bin > 0, 2] * 0.5

            # 創建並保存圖表
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_resized)
            axes[0].set_title('原始影像')
            axes[0].axis('off')

            axes[1].imshow(pred.squeeze(), cmap='jet')
            axes[1].set_title(f'預測熱圖 (得分: {error_score:.3f})')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title('疊加效果')
            axes[2].axis('off')

            if 'error_label' in row:
                fig.suptitle(f"預測: {error_status}, 真實: {true_status}, 正確: {correct}")
            else:
                fig.suptitle(f"預測: {error_status}")

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
                threshold=config.THRESHOLD
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

    parser = argparse.ArgumentParser(description='3D列印錯誤偵測預測腳本')
    parser.add_argument('--model', type=str, required=True, help='訓練好的模型路徑')
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
    model = load_trained_model(args.model)

    # 根據模式進行預測
    if args.mode == 'validation':
        print("使用驗證集進行預測...")

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
        results = evaluate_model(
            model=model,
            test_generator=val_generator,
            num_samples=len(val_generator) * config.BATCH_SIZE,
            threshold=config.THRESHOLD,
            save_path=os.path.join(args.output, "evaluation_results.json")
        )

        # 視覺化預測結果
        os.makedirs(os.path.join(args.output, "visualizations"), exist_ok=True)
        visualize_predictions(
            model=model,
            test_generator=val_generator,
            num_samples=10,
            threshold=config.THRESHOLD,
            save_dir=os.path.join(args.output, "visualizations")
        )

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