"""
訓練腳本
用於訓練3D列印錯誤偵測模型
"""

import os
import time
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import pandas as pd
import matplotlib.pyplot as plt

from config.config import Config
from models.unet import get_unet_with_custom_params
from utils.dataset import load_dataset_from_indices
from utils.metrics import dice_coefficient, iou_coefficient
from utils.visualization import plot_training_history, visualize_predictions_grid


def train_model():
    """
    根據配置訓練模型
    """
    # 載入配置
    config = Config()

    print(f"開始訓練 {config.MODEL_NAME} 模型")
    print(f"配置參數:")
    print(f"  - 輸入尺寸: {config.INPUT_SIZE}")
    print(f"  - 批次大小: {config.BATCH_SIZE}")
    print(f"  - 訓練輪數: {config.EPOCHS}")
    print(f"  - 學習率: {config.LEARNING_RATE}")
    print(f"  - 損失函數: {config.LOSS_TYPE}")

    # 設置GPU記憶體增長
    if config.GPU_MEMORY_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"已啟用GPU記憶體增長")

    # 建立結果目錄
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(config.RESULT_DIR, f"{config.MODEL_NAME}_{timestamp}")
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    log_dir = os.path.join(result_dir, "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 載入資料
    print("載入訓練與驗證資料...")

    train_indices_file = os.path.join(config.DATA_DIR, "processed", "train_indices.csv")
    val_indices_file = os.path.join(config.DATA_DIR, "processed", "val_indices.csv")

    train_dir = os.path.join(config.DATA_DIR, "processed", "train")
    val_dir = os.path.join(config.DATA_DIR, "processed", "val")

    train_generator = load_dataset_from_indices(
        processed_dir=train_dir,
        indices_file=train_indices_file,
        batch_size=config.BATCH_SIZE,
        is_training=True,
        input_size=config.INPUT_SIZE[:2]
    )

    val_generator = load_dataset_from_indices(
        processed_dir=val_dir,
        indices_file=val_indices_file,
        batch_size=config.BATCH_SIZE,
        is_training=False,
        input_size=config.INPUT_SIZE[:2]
    )

    print(f"訓練資料生成器長度: {len(train_generator)}")
    print(f"驗證資料生成器長度: {len(val_generator)}")

    # 定義訓練指標
    metrics = [
        dice_coefficient,
        iou_coefficient,
        'accuracy'
    ]

    # 建立模型
    print("建立模型...")
    config.METRICS = metrics
    model = get_unet_with_custom_params(config)

    model.summary()

    # 設置回調函數
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_epoch{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            update_freq='epoch'
        )
    ]

    # 訓練模型
    print("開始訓練...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    # 保存訓練歷史
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(result_dir, 'training_history.csv'), index=False)

    # 繪製訓練曲線
    plot_training_history(history, os.path.join(result_dir, 'training_curves.png'))

    # 在驗證集上可視化一些預測結果
    print("視覺化模型預測結果...")

    # 取一個批次的資料進行視覺化
    val_images, val_masks = next(iter(val_generator))

    visualize_predictions_grid(
        model=model,
        test_images=val_images,
        test_masks=val_masks,
        num_samples=6,
        threshold=config.THRESHOLD,
        output_path=os.path.join(result_dir, 'predictions_visualization.png')
    )

    # 保存最終模型
    final_model_path = os.path.join(result_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"模型已保存至 {final_model_path}")

    # 保存配置
    config_path = os.path.join(result_dir, 'model_config.txt')
    with open(config_path, 'w') as f:
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key} = {value}\n")

    print(f"訓練完成。結果保存至: {result_dir}")
    return model, history, result_dir


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()