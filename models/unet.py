"""
U-Net 模型定義
適用於3D列印錯誤偵測的影像分割任務

本模組實現了經典的U-Net架構，包含編碼器下採樣路徑和解碼器上採樣路徑
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate,
    BatchNormalization, Activation
)


def conv_block(input_tensor, num_filters, kernel_size=3, batchnorm=True):
    """
    標準卷積區塊：兩個卷積層加上批次正規化（可選）

    Args:
        input_tensor: 輸入張量
        num_filters: 卷積濾波器數量
        kernel_size: 卷積核大小
        batchnorm: 是否使用批次正規化

    Returns:
        x: 經過卷積處理後的張量
    """
    # 第一個卷積層
    x = Conv2D(num_filters, (kernel_size, kernel_size),
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第二個卷積層
    x = Conv2D(num_filters, (kernel_size, kernel_size),
               padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def unet_model(input_size=(224, 224, 3), num_classes=1, base_filters=32,
               depth=4, dropout_rate=0.1, batchnorm=True):
    """
    構建完整的U-Net模型

    Args:
        input_size: 輸入影像尺寸，預設為(224, 224, 3)
        num_classes: 輸出分類數量，二元分割為1，多類別分割為n
        base_filters: 最初的過濾器數量，預設為32
        depth: U-Net深度（下採樣次數），預設為4
        dropout_rate: Dropout比率，用於防止過擬合
        batchnorm: 是否使用批次正規化

    Returns:
        model: 完整配置的U-Net模型
    """
    inputs = Input(input_size)

    # 儲存編碼器各層輸出，用於之後的跳接連結
    skip_connections = []

    # === 編碼器路徑 ===
    x = inputs
    for i in range(depth):
        x = conv_block(x, base_filters * (2 ** i), batchnorm=batchnorm)
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_rate)(x)

    # === 瓶頸 ===
    x = conv_block(x, base_filters * (2 ** depth), batchnorm=batchnorm)

    # === 解碼器路徑 ===
    for i in reversed(range(depth)):
        # 上採樣
        x = UpSampling2D((2, 2))(x)
        # 從編碼器獲取跳接連結
        skip = skip_connections[i]
        # 拼接跳接連結
        x = concatenate([x, skip], axis=3)
        x = Dropout(dropout_rate)(x)
        x = conv_block(x, base_filters * (2 ** i), batchnorm=batchnorm)

    # === 輸出層 ===
    if num_classes == 1:  # 二元分割
        output = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    else:  # 多類別分割
        output = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # 構建模型
    model = Model(inputs=[inputs], outputs=[output])

    return model


def get_unet_with_custom_params(config):
    """
    根據配置參數構建U-Net模型

    Args:
        config: 包含模型參數的配置對象

    Returns:
        model: 配置的U-Net模型
    """
    model = unet_model(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        base_filters=config.BASE_FILTERS,
        depth=config.DEPTH,
        dropout_rate=config.DROPOUT_RATE,
        batchnorm=config.USE_BATCHNORM
    )

    # 根據配置選擇損失函數
    if config.LOSS_TYPE == 'bce':
        loss = tf.keras.losses.BinaryCrossentropy()
    elif config.LOSS_TYPE == 'dice':
        loss = dice_loss
    elif config.LOSS_TYPE == 'bce_dice':
        loss = combined_loss
    else:
        raise ValueError(f"不支援的損失函數類型: {config.LOSS_TYPE}")

    # 配置模型編譯參數
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=config.METRICS
    )

    return model


def dice_coef(y_true, y_pred, smooth=1.0):
    """
    計算Dice係數

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        smooth: 平滑因子，避免除以零

    Returns:
        dice: Dice係數值
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice損失函數，用於直接優化Dice係數

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤

    Returns:
        loss: 1 - Dice係數
    """
    return 1.0 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred, alpha=0.5):
    """
    結合Binary Cross-Entropy和Dice損失的混合損失函數

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        alpha: BCE損失的權重，Dice損失的權重為(1-alpha)

    Returns:
        loss: 加權的混合損失值
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dice


def iou_coef(y_true, y_pred, smooth=1.0):
    """
    計算IoU係數 (Intersection over Union)

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        smooth: 平滑因子，避免除以零

    Returns:
        iou: IoU係數值
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)