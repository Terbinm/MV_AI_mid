"""
訓練腳本 - 分類器版本
用於訓練3D列印錯誤分類模型
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
from models.unet import get_unet_with_custom_params  # 使用修改後的函數
from utils.dataset import load_dataset_for_classification  # 使用新的資料載入器
from utils.visualization import plot_training_history, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_classifier_model():
    """
    根據配置訓練二元分類模型
    """
    # 載入配置
    config = Config()

    print(f"開始訓練 {config.MODEL_NAME} 分類器模型")
    print(f"配置參數:")
    print(f"  - 輸入尺寸: {config.INPUT_SIZE}")
    print(f"  - 批次大小: {config.BATCH_SIZE}")
    print(f"  - 訓練輪數: {config.EPOCHS}")
    print(f"  - 學習率: {config.LEARNING_RATE}")

    # 設置GPU記憶體增長
    if config.GPU_MEMORY_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"已啟用GPU記憶體增長")

    # 建立結果目錄
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.join(config.RESULT_DIR, f"{config.MODEL_NAME}_classifier_{timestamp}")
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

    train_generator = load_dataset_for_classification(
        processed_dir=train_dir,
        indices_file=train_indices_file,
        batch_size=config.BATCH_SIZE,
        is_training=True,
        input_size=config.INPUT_SIZE[:2]
    )

    val_generator = load_dataset_for_classification(
        processed_dir=val_dir,
        indices_file=val_indices_file,
        batch_size=config.BATCH_SIZE,
        is_training=False,
        input_size=config.INPUT_SIZE[:2]
    )

    print(f"訓練資料生成器長度: {len(train_generator)}")
    print(f"驗證資料生成器長度: {len(val_generator)}")

    # 定義訓練指標 - 分類指標
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    # 建立模型 - 使用分類器模式
    print("建立分類器模型...")
    config.METRICS = metrics
    model = get_unet_with_custom_params(config, is_classifier=True)

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
            monitor='val_accuracy',
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
    print("開始訓練分類器...")
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

    # 在驗證集上評估模型
    print("評估模型在驗證集上的表現...")

    # 收集所有驗證集預測
    y_true = []
    y_pred = []

    for i in range(len(val_generator)):
        x_batch, y_batch = val_generator[i]
        predictions = model.predict(x_batch)

        y_true.extend(y_batch)
        y_pred.extend((predictions > 0.5).astype(int).flatten())

    # 計算評估指標
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    # 混淆矩陣
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred)
    print("\n混淆矩陣:")
    print(cm)

    # 繪製混淆矩陣
    plot_confusion_matrix(cm, ['正常', '錯誤'], os.path.join(result_dir, 'confusion_matrix.png'))

    # 分類報告
    report = classification_report(y_true, y_pred, target_names=['正常', '錯誤'])
    print("\n分類報告:")
    print(report)

    with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

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
        train_classifier_model()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()