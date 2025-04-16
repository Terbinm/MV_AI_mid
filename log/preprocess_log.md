(tf_gpu) PS C:\led_code\MV_AI_mid_ssd> python main.py
已設定中文字體: C:/Windows/Fonts/msjh.ttc

    =======================================================
      3D列印錯誤偵測系統 - 基於影像分割的自動錯誤檢測
    =======================================================


    請選擇操作:
    1. 資料預處理
    2. 模型訓練
    3. 模型評估
    4. 錯誤預測
    5. 觀察資料分布
    6. 測試資料平衡策略
    7. 退出

請輸入選項編號: 1
正在執行資料預處理...
C:\Users\user\anaconda3\envs\tf_gpu\lib\site-packages\albumentations\__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.5 (you have 1.4.18). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
是否清理先前處理的資料? [y/N]: y
正在刪除處理後的資料...
已成功刪除目錄: C:\led_code\MV_AI_mid_ssd\data\processed
是否自訂資料平衡參數? [y/N]: y
設置錯誤樣本比例 (0.0-1.0) [預設: 0.5]:
使用預設值
選擇平衡策略 (undersample/oversample/hybrid) [預設: hybrid]:
設置最大樣本數 [預設: 10000]:
使用預設值
開始資料預處理流程...
分析CSV資料: C:\led_code\MV_AI_mid_ssd\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv
資料總筆數: 156539
欄位數量: 18
欄位名稱: img_path, timestamp, flow_rate, feed_rate, z_offset, target_hotend, hotend, bed, nozzle_tip_x, nozzle_tip_y, img_num, print_id, flow_rate_class, feed_rate_class, z_offset_class, hotend_class, img_mean, img_std

flow_rate_class 分佈:
  值=0: 56684 筆 (36.21%)
  值=1: 38088 筆 (24.33%)
  值=2: 61767 筆 (39.46%)

feed_rate_class 分佈:
  值=0: 56740 筆 (36.25%)
  值=1: 38953 筆 (24.88%)
  值=2: 60846 筆 (38.87%)

z_offset_class 分佈:
  值=0: 18793 筆 (12.01%)
  值=1: 39215 筆 (25.05%)
  值=2: 98531 筆 (62.94%)

hotend_class 分佈:
  值=0: 48682 筆 (31.10%)
  值=1: 84270 筆 (53.83%)
  值=2: 23587 筆 (15.07%)

獨特列印任務數量: 30
每個列印任務的平均影像數: 5217.97

影像平均值的平均: 64.6224
影像標準差的平均: 40.4800

錯誤統計:
  正常樣本: 1940 (1.24%)
  錯誤樣本: 154599 (98.76%)

錯誤類型分佈:
  flow_feed_z_offset: 34094 (21.78%)
  flow_feed_z_offset_hotend: 33046 (21.11%)
  feed_z_offset: 12755 (8.15%)
  flow_feed: 12004 (7.67%)
  flow_z_offset: 11393 (7.28%)
  flow_z_offset_hotend: 10890 (6.96%)
  flow_feed_hotend: 10156 (6.49%)
  feed_z_offset_hotend: 8319 (5.31%)
  flow: 4083 (2.61%)
  z_offset: 4025 (2.57%)
  feed: 3976 (2.54%)
  feed_hotend: 3236 (2.07%)
  z_offset_hotend: 2802 (1.79%)
  flow_hotend: 2785 (1.78%)
  normal: 1940 (1.24%)
  hotend: 1035 (0.66%)
檢查影像路徑: C:\led_code\MV_AI_mid_ssd\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv
檢查影像: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 156539/156539 [00:06<00:00, 25136.21it/s]
檢查完成: 共 156539 個有效影像, 0 個缺失影像
抽樣並探索影像: C:\led_code\MV_AI_mid_ssd\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv
處理正常樣本...
Corrupt JPEG data: 1 extraneous bytes before marker 0xd1
Corrupt JPEG data: 1 extraneous bytes before marker 0xd5
Corrupt JPEG data: 6 extraneous bytes before marker 0xd5
Corrupt JPEG data: 2 extraneous bytes before marker 0xd4
Corrupt JPEG data: 6 extraneous bytes before marker 0xd2
處理錯誤樣本...
Corrupt JPEG data: 1 extraneous bytes before marker 0xd6
Corrupt JPEG data: 7 extraneous bytes before marker 0xd4
Corrupt JPEG data: 2 extraneous bytes before marker 0xd6
Corrupt JPEG data: 1 extraneous bytes before marker 0xd6
Corrupt JPEG data: 3 extraneous bytes before marker 0xd1
已保存 5 個正常樣本和 5 個錯誤樣本到 C:\led_code\MV_AI_mid_ssd\data\plt\sample_images
創建平衡樣本: C:\led_code\MV_AI_mid_ssd\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv -> C:\led_code\MV_AI_mid_ssd\data\processed\balanced_dataset.csv
平衡資料集: C:\led_code\MV_AI_mid_ssd\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv -> C:\led_code\MV_AI_mid_ssd\data\processed\balanced_dataset.csv
使用策略: hybrid, 平衡比例: 0.5, 最大樣本數: 10000
原始資料集: 156539 筆
原始分布:
  - 正常樣本: 1940 (1.24%)
  - 錯誤樣本: 154599 (98.76%)
平衡後分布 (策略: hybrid):
  - 正常樣本: 5000 (50.00%)
  - 錯誤樣本: 5000 (50.00%)
  - 總樣本數: 10000
平衡後的資料集已保存至: C:\led_code\MV_AI_mid_ssd\data\processed\balanced_dataset.csv
載入資料集: C:\led_code\MV_AI_mid_ssd\data\processed\balanced_dataset.csv
載入完成，共 10000 筆資料
分割訓練集和驗證集...
訓練集大小: 8000, 驗證集大小: 2000
索引檔案已保存至 C:\led_code\MV_AI_mid_ssd\data\processed\train_indices.csv 和 C:\led_code\MV_AI_mid_ssd\data\processed\val_indices.csv
訓練集分布: 正常=4000 (50.00%), 錯誤=4000 (50.00%)
驗證集分布: 正常=1000 (50.00%), 錯誤=1000 (50.00%)
預處理訓練集影像...
Corrupt JPEG data: 2 extraneous bytes before marker 0xd2
Corrupt JPEG data: 3 extraneous bytes before marker 0xd4
Corrupt JPEG data: 6 extraneous bytes before marker 0xd4
...
Corrupt JPEG data: 2 extraneous bytes before marker 0xd6
資料預處理完成！
資料預處理完成!

請選擇操作:
    1. 資料預處理
    2. 模型訓練
    3. 模型評估
    4. 錯誤預測
    5. 觀察資料分布
    6. 測試資料平衡策略
    7. 退出

請輸入選項編號: 2
正在執行模型訓練...
C:\Users\user\anaconda3\envs\tf_gpu\lib\site-packages\albumentations\__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.5 (you have 1.4.18). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
是否自訂訓練參數? [y/N]: n
開始訓練 unet_20250415_0656 模型
配置參數:
  - 輸入尺寸: (224, 224, 3)
  - 批次大小: 16
  - 訓練輪數: 50
  - 學習率: 0.0001
  - 損失函數: bce_dice
已啟用GPU記憶體增長
載入訓練與驗證資料...
訓練資料生成器長度: 500
驗證資料生成器長度: 125
建立模型...
2025-04-15 06:56:02.800246: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-04-15 06:56:03.193729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21344 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:01:00.0, compute capability: 8.9
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 224, 224, 3  0           []
                                )]

 conv2d (Conv2D)                (None, 224, 224, 32  896         ['input_1[0][0]']
                                )

 batch_normalization (BatchNorm  (None, 224, 224, 32  128        ['conv2d[0][0]']
 alization)                     )

 activation (Activation)        (None, 224, 224, 32  0           ['batch_normalization[0][0]']
                                )

 conv2d_1 (Conv2D)              (None, 224, 224, 32  9248        ['activation[0][0]']
                                )

 batch_normalization_1 (BatchNo  (None, 224, 224, 32  128        ['conv2d_1[0][0]']
 rmalization)                   )

 activation_1 (Activation)      (None, 224, 224, 32  0           ['batch_normalization_1[0][0]']
                                )

 max_pooling2d (MaxPooling2D)   (None, 112, 112, 32  0           ['activation_1[0][0]']
                                )

 dropout (Dropout)              (None, 112, 112, 32  0           ['max_pooling2d[0][0]']
                                )

 conv2d_2 (Conv2D)              (None, 112, 112, 64  18496       ['dropout[0][0]']
                                )

 batch_normalization_2 (BatchNo  (None, 112, 112, 64  256        ['conv2d_2[0][0]']
 rmalization)                   )

 activation_2 (Activation)      (None, 112, 112, 64  0           ['batch_normalization_2[0][0]']
                                )

 conv2d_3 (Conv2D)              (None, 112, 112, 64  36928       ['activation_2[0][0]']
                                )

 batch_normalization_3 (BatchNo  (None, 112, 112, 64  256        ['conv2d_3[0][0]']
 rmalization)                   )

 activation_3 (Activation)      (None, 112, 112, 64  0           ['batch_normalization_3[0][0]']
                                )

 max_pooling2d_1 (MaxPooling2D)  (None, 56, 56, 64)  0           ['activation_3[0][0]']

 dropout_1 (Dropout)            (None, 56, 56, 64)   0           ['max_pooling2d_1[0][0]']

 conv2d_4 (Conv2D)              (None, 56, 56, 128)  73856       ['dropout_1[0][0]']

 batch_normalization_4 (BatchNo  (None, 56, 56, 128)  512        ['conv2d_4[0][0]']
 rmalization)

 activation_4 (Activation)      (None, 56, 56, 128)  0           ['batch_normalization_4[0][0]']

 conv2d_5 (Conv2D)              (None, 56, 56, 128)  147584      ['activation_4[0][0]']

 batch_normalization_5 (BatchNo  (None, 56, 56, 128)  512        ['conv2d_5[0][0]']
 rmalization)

 activation_5 (Activation)      (None, 56, 56, 128)  0           ['batch_normalization_5[0][0]']

 max_pooling2d_2 (MaxPooling2D)  (None, 28, 28, 128)  0          ['activation_5[0][0]']

 dropout_2 (Dropout)            (None, 28, 28, 128)  0           ['max_pooling2d_2[0][0]']

 conv2d_6 (Conv2D)              (None, 28, 28, 256)  295168      ['dropout_2[0][0]']

 batch_normalization_6 (BatchNo  (None, 28, 28, 256)  1024       ['conv2d_6[0][0]']
 rmalization)

 activation_6 (Activation)      (None, 28, 28, 256)  0           ['batch_normalization_6[0][0]']

 conv2d_7 (Conv2D)              (None, 28, 28, 256)  590080      ['activation_6[0][0]']

 batch_normalization_7 (BatchNo  (None, 28, 28, 256)  1024       ['conv2d_7[0][0]']
 rmalization)

 activation_7 (Activation)      (None, 28, 28, 256)  0           ['batch_normalization_7[0][0]']

 max_pooling2d_3 (MaxPooling2D)  (None, 14, 14, 256)  0          ['activation_7[0][0]']

 dropout_3 (Dropout)            (None, 14, 14, 256)  0           ['max_pooling2d_3[0][0]']

 conv2d_8 (Conv2D)              (None, 14, 14, 512)  1180160     ['dropout_3[0][0]']

 batch_normalization_8 (BatchNo  (None, 14, 14, 512)  2048       ['conv2d_8[0][0]']
 rmalization)

 activation_8 (Activation)      (None, 14, 14, 512)  0           ['batch_normalization_8[0][0]']

 conv2d_9 (Conv2D)              (None, 14, 14, 512)  2359808     ['activation_8[0][0]']

 batch_normalization_9 (BatchNo  (None, 14, 14, 512)  2048       ['conv2d_9[0][0]']
 rmalization)

 activation_9 (Activation)      (None, 14, 14, 512)  0           ['batch_normalization_9[0][0]']

 up_sampling2d (UpSampling2D)   (None, 28, 28, 512)  0           ['activation_9[0][0]']

 concatenate (Concatenate)      (None, 28, 28, 768)  0           ['up_sampling2d[0][0]',
                                                                  'activation_7[0][0]']

 dropout_4 (Dropout)            (None, 28, 28, 768)  0           ['concatenate[0][0]']

 conv2d_10 (Conv2D)             (None, 28, 28, 256)  1769728     ['dropout_4[0][0]']

 batch_normalization_10 (BatchN  (None, 28, 28, 256)  1024       ['conv2d_10[0][0]']
 ormalization)

 activation_10 (Activation)     (None, 28, 28, 256)  0           ['batch_normalization_10[0][0]']

 conv2d_11 (Conv2D)             (None, 28, 28, 256)  590080      ['activation_10[0][0]']

 batch_normalization_11 (BatchN  (None, 28, 28, 256)  1024       ['conv2d_11[0][0]']
 ormalization)

 activation_11 (Activation)     (None, 28, 28, 256)  0           ['batch_normalization_11[0][0]']

 up_sampling2d_1 (UpSampling2D)  (None, 56, 56, 256)  0          ['activation_11[0][0]']

 concatenate_1 (Concatenate)    (None, 56, 56, 384)  0           ['up_sampling2d_1[0][0]',
                                                                  'activation_5[0][0]']

 dropout_5 (Dropout)            (None, 56, 56, 384)  0           ['concatenate_1[0][0]']

 conv2d_12 (Conv2D)             (None, 56, 56, 128)  442496      ['dropout_5[0][0]']

 batch_normalization_12 (BatchN  (None, 56, 56, 128)  512        ['conv2d_12[0][0]']
 ormalization)

 activation_12 (Activation)     (None, 56, 56, 128)  0           ['batch_normalization_12[0][0]']

 conv2d_13 (Conv2D)             (None, 56, 56, 128)  147584      ['activation_12[0][0]']

 batch_normalization_13 (BatchN  (None, 56, 56, 128)  512        ['conv2d_13[0][0]']
 ormalization)

 activation_13 (Activation)     (None, 56, 56, 128)  0           ['batch_normalization_13[0][0]']

 up_sampling2d_2 (UpSampling2D)  (None, 112, 112, 12  0          ['activation_13[0][0]']
                                8)

 concatenate_2 (Concatenate)    (None, 112, 112, 19  0           ['up_sampling2d_2[0][0]',
                                2)                                'activation_3[0][0]']

 dropout_6 (Dropout)            (None, 112, 112, 19  0           ['concatenate_2[0][0]']
                                2)

 conv2d_14 (Conv2D)             (None, 112, 112, 64  110656      ['dropout_6[0][0]']
                                )

 batch_normalization_14 (BatchN  (None, 112, 112, 64  256        ['conv2d_14[0][0]']
 ormalization)                  )

 activation_14 (Activation)     (None, 112, 112, 64  0           ['batch_normalization_14[0][0]']
                                )

 conv2d_15 (Conv2D)             (None, 112, 112, 64  36928       ['activation_14[0][0]']
                                )

 batch_normalization_15 (BatchN  (None, 112, 112, 64  256        ['conv2d_15[0][0]']
 ormalization)                  )

 activation_15 (Activation)     (None, 112, 112, 64  0           ['batch_normalization_15[0][0]']
                                )

 up_sampling2d_3 (UpSampling2D)  (None, 224, 224, 64  0          ['activation_15[0][0]']
                                )

 concatenate_3 (Concatenate)    (None, 224, 224, 96  0           ['up_sampling2d_3[0][0]',
                                )                                 'activation_1[0][0]']

 dropout_7 (Dropout)            (None, 224, 224, 96  0           ['concatenate_3[0][0]']
                                )

 conv2d_16 (Conv2D)             (None, 224, 224, 32  27680       ['dropout_7[0][0]']
                                )

 batch_normalization_16 (BatchN  (None, 224, 224, 32  128        ['conv2d_16[0][0]']
 ormalization)                  )

 activation_16 (Activation)     (None, 224, 224, 32  0           ['batch_normalization_16[0][0]']
                                )

 conv2d_17 (Conv2D)             (None, 224, 224, 32  9248        ['activation_16[0][0]']
                                )

 batch_normalization_17 (BatchN  (None, 224, 224, 32  128        ['conv2d_17[0][0]']
 ormalization)                  )

 activation_17 (Activation)     (None, 224, 224, 32  0           ['batch_normalization_17[0][0]']
                                )

 conv2d_18 (Conv2D)             (None, 224, 224, 1)  33          ['activation_17[0][0]']

==================================================================================================
Total params: 7,858,433
Trainable params: 7,852,545
Non-trainable params: 5,888
__________________________________________________________________________________________________
開始訓練...
Epoch 1/50
2025-04-15 06:56:05.193978: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
2025-04-15 06:56:06.448841: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
500/500 [==============================] - ETA: 0s - loss: 0.5575 - dice_coefficient: 0.5598 - iou_coefficient: 0.3932 - accuracy: 0.6128
Epoch 1: val_loss improved from inf to 0.54378, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\model_epoch01_val_loss_0.5438.h5

Epoch 1: val_dice_coefficient improved from -inf to 0.60332, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5
500/500 [==============================] - 53s 98ms/step - loss: 0.5575 - dice_coefficient: 0.5598 - iou_coefficient: 0.3932 - accuracy: 0.6128 - val_loss: 0.5438 - val_dice_coefficient: 0.6033 - val_iou_coefficient: 0.4384 - val_accuracy: 0.5601 - lr: 1.0000e-04
Epoch 2/50
500/500 [==============================] - ETA: 0s - loss: 0.5201 - dice_coefficient: 0.5982 - iou_coefficient: 0.4311 - accuracy: 0.6513
Epoch 2: val_loss improved from 0.54378 to 0.51916, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\model_epoch02_val_loss_0.5192.h5

Epoch 2: val_dice_coefficient improved from 0.60332 to 0.62057, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5
500/500 [==============================] - 44s 89ms/step - loss: 0.5201 - dice_coefficient: 0.5982 - iou_coefficient: 0.4311 - accuracy: 0.6513 - val_loss: 0.5192 - val_dice_coefficient: 0.6206 - val_iou_coefficient: 0.4564 - val_accuracy: 0.6133 - lr: 1.0000e-04
Epoch 3/50
500/500 [==============================] - ETA: 0s - loss: 0.5019 - dice_coefficient: 0.6154 - iou_coefficient: 0.4486 - accuracy: 0.6644
Epoch 3: val_loss improved from 0.51916 to 0.48183, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\model_epoch03_val_loss_0.4818.h5

Epoch 3: val_dice_coefficient improved from 0.62057 to 0.62582, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5
500/500 [==============================] - 45s 89ms/step - loss: 0.5019 - dice_coefficient: 0.6154 - iou_coefficient: 0.4486 - accuracy: 0.6644 - val_loss: 0.4818 - val_dice_coefficient: 0.6258 - val_iou_coefficient: 0.4617 - val_accuracy: 0.6745 - lr: 1.0000e-04
Epoch 4/50
500/500 [==============================] - ETA: 0s - loss: 0.4878 - dice_coefficient: 0.6267 - iou_coefficient: 0.4612 - accuracy: 0.6745
Epoch 4: val_loss improved from 0.48183 to 0.45941, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\model_epoch04_val_loss_0.4594.h5

Epoch 4: val_dice_coefficient improved from 0.62582 to 0.65191, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5
500/500 [==============================] - 44s 89ms/step - loss: 0.4878 - dice_coefficient: 0.6267 - iou_coefficient: 0.4612 - accuracy: 0.6745 - val_loss: 0.4594 - val_dice_coefficient: 0.6519 - val_iou_coefficient: 0.4900 - val_accuracy: 0.7128 - lr: 1.0000e-04
Epoch 5/50
500/500 [==============================] - ETA: 0s - loss: 0.4812 - dice_coefficient: 0.6336 - iou_coefficient: 0.4684 - accuracy: 0.6801
Epoch 5: val_loss improved from 0.45941 to 0.45268, saving model to C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\checkpoints\model_epoch05_val_loss_0.4527.h5
...
Epoch 48: val_dice_coefficient did not improve from 0.81739
500/500 [==============================] - 45s 89ms/step - loss: 0.2586 - dice_coefficient: 0.8115 - iou_coefficient: 0.6869 - accuracy: 0.8600 - val_loss: 0.2639 - val_dice_coefficient: 0.7987 - val_iou_coefficient: 0.6706 - val_accuracy: 0.8596 - lr: 1.0000e-04
Epoch 49/50
500/500 [==============================] - ETA: 0s - loss: 0.2580 - dice_coefficient: 0.8118 - iou_coefficient: 0.6883 - accuracy: 0.8606
Epoch 49: val_loss did not improve from 0.24010

Epoch 49: val_dice_coefficient did not improve from 0.81739
500/500 [==============================] - 48s 96ms/step - loss: 0.2580 - dice_coefficient: 0.8118 - iou_coefficient: 0.6883 - accuracy: 0.8606 - val_loss: 0.2496 - val_dice_coefficient: 0.8131 - val_iou_coefficient: 0.6910 - val_accuracy: 0.8724 - lr: 1.0000e-04
Epoch 50/50
500/500 [==============================] - ETA: 0s - loss: 0.2548 - dice_coefficient: 0.8145 - iou_coefficient: 0.6917 - accuracy: 0.8627
Epoch 50: val_loss did not improve from 0.24010

Epoch 50: val_dice_coefficient did not improve from 0.81739
500/500 [==============================] - 44s 89ms/step - loss: 0.2548 - dice_coefficient: 0.8145 - iou_coefficient: 0.6917 - accuracy: 0.8627 - val_loss: 0.2485 - val_dice_coefficient: 0.8143 - val_iou_coefficient: 0.6922 - val_accuracy: 0.8729 - lr: 1.0000e-04
訓練歷史曲線已保存至 C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\training_curves.png
視覺化模型預測結果...
1/1 [==============================] - 0s 183ms/step
1/1 [==============================] - 0s 11ms/step
1/1 [==============================] - 0s 12ms/step
1/1 [==============================] - 0s 12ms/step
1/1 [==============================] - 0s 11ms/step
1/1 [==============================] - 0s 11ms/step
預測視覺化已保存至 C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\predictions_visualization.png
模型已保存至 C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602\final_model.h5
訓練完成。結果保存至: C:\led_code\MV_AI_mid_ssd\results\unet_20250415_0656_20250415-065602
模型訓練程序完成!

按Enter返回主選單...

    請選擇操作:
    1. 資料預處理
    2. 模型訓練
    3. 模型評估
    4. 錯誤預測
    5. 觀察資料分布
    6. 測試資料平衡策略
    7. 退出

請輸入選項編號: 3
正在執行模型評估...

可用模型:
1. final_model.h5
2. best_model.h5
3. model_epoch01_val_loss_0.5438.h5
4. model_epoch02_val_loss_0.5192.h5
5. model_epoch03_val_loss_0.4818.h5
6. model_epoch04_val_loss_0.4594.h5
7. model_epoch05_val_loss_0.4527.h5
8. model_epoch07_val_loss_0.4250.h5
9. model_epoch08_val_loss_0.4209.h5
10. model_epoch09_val_loss_0.4191.h5
11. model_epoch10_val_loss_0.4078.h5
12. model_epoch11_val_loss_0.4010.h5
13. model_epoch12_val_loss_0.4009.h5
14. model_epoch13_val_loss_0.3802.h5
15. model_epoch14_val_loss_0.3758.h5
16. model_epoch17_val_loss_0.3708.h5
17. model_epoch18_val_loss_0.3625.h5
18. model_epoch19_val_loss_0.3548.h5
19. model_epoch20_val_loss_0.3508.h5
20. model_epoch21_val_loss_0.3471.h5
21. model_epoch22_val_loss_0.3404.h5
22. model_epoch26_val_loss_0.3328.h5
23. model_epoch27_val_loss_0.3235.h5
24. model_epoch28_val_loss_0.3135.h5
25. model_epoch31_val_loss_0.3021.h5
26. model_epoch32_val_loss_0.2957.h5
27. model_epoch35_val_loss_0.2758.h5
28. model_epoch38_val_loss_0.2687.h5
29. model_epoch42_val_loss_0.2606.h5
30. model_epoch43_val_loss_0.2584.h5
31. model_epoch46_val_loss_0.2401.h5

請選擇模型編號: 2

評估模式:
1. 驗證集評估
2. 自定義CSV評估

請選擇評估模式: 1
是否自訂二值化閾值? [y/N]: n

執行命令: python evaluate.py --model results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5 --mode validation --output results\evaluation\eval_best_model
C:\Users\user\anaconda3\envs\tf_gpu\lib\site-packages\albumentations\__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.5 (you have 1.4.18). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
已設定中文字體: C:/Windows/Fonts/msjh.ttc
已啟用GPU記憶體增長
載入模型: results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5
2025-04-15 07:34:33.702735: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-04-15 07:34:34.080554: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21344 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:01:00.0, compute capability: 8.9
模型載入成功
在驗證集上評估模型...
開始評估模型...
2025-04-15 07:34:35.324327: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
2025-04-15 07:34:36.539833: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
1/1 [==============================] - 2s 2s/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 13ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 15ms/step
...
1/1 [==============================] - 0s 17ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 15ms/step
像素準確率: 0.8748
Dice係數: 0.8749
IoU係數: 0.7777

圖像級別的分類性能:
準確率: 0.7745
敏感度 (召回率): 0.9880
特異度: 0.5610
精確度: 0.6924
混淆矩陣已保存至 results\evaluation\eval_best_model\confusion_matrix.png
評估結果已保存至 results\evaluation\eval_best_model\validation_results.json
1/1 [==============================] - 0s 14ms/step
評估不同閾值下的模型性能...
1/1 [==============================] - 0s 16ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 15ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 13ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 13ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 15ms/step
1/1 [==============================] - 0s 14ms/step
最佳F1分數 (0.9024) 對應的閾值為: 0.45000000000000007
評估完成，結果已保存至: results\evaluation\eval_best_model
模型評估完成! 結果保存在: results\evaluation\eval_best_model

按Enter返回主選單...

    請選擇操作:
    1. 資料預處理
    2. 模型訓練
    3. 模型評估
    4. 錯誤預測
    5. 觀察資料分布
    6. 測試資料平衡策略
    7. 退出

請輸入選項編號: 4
正在執行錯誤預測...

可用模型:
1. final_model.h5
2. best_model.h5
3. model_epoch01_val_loss_0.5438.h5
4. model_epoch02_val_loss_0.5192.h5
5. model_epoch03_val_loss_0.4818.h5
6. model_epoch04_val_loss_0.4594.h5
7. model_epoch05_val_loss_0.4527.h5
8. model_epoch07_val_loss_0.4250.h5
9. model_epoch08_val_loss_0.4209.h5
10. model_epoch09_val_loss_0.4191.h5
11. model_epoch10_val_loss_0.4078.h5
12. model_epoch11_val_loss_0.4010.h5
13. model_epoch12_val_loss_0.4009.h5
14. model_epoch13_val_loss_0.3802.h5
15. model_epoch14_val_loss_0.3758.h5
16. model_epoch17_val_loss_0.3708.h5
17. model_epoch18_val_loss_0.3625.h5
18. model_epoch19_val_loss_0.3548.h5
19. model_epoch20_val_loss_0.3508.h5
20. model_epoch21_val_loss_0.3471.h5
21. model_epoch22_val_loss_0.3404.h5
22. model_epoch26_val_loss_0.3328.h5
23. model_epoch27_val_loss_0.3235.h5
24. model_epoch28_val_loss_0.3135.h5
25. model_epoch31_val_loss_0.3021.h5
26. model_epoch32_val_loss_0.2957.h5
27. model_epoch35_val_loss_0.2758.h5
28. model_epoch38_val_loss_0.2687.h5
29. model_epoch42_val_loss_0.2606.h5
30. model_epoch43_val_loss_0.2584.h5
31. model_epoch46_val_loss_0.2401.h5

請選擇模型編號: 2

預測模式:
1. 驗證集預測
2. 目錄預測 (預測指定目錄中的所有影像)
3. CSV預測 (根據CSV檔案中的路徑預測)

請選擇預測模式: 1
是否自訂二值化閾值? [y/N]: n

執行命令: python predict.py --model results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5 --mode validation --output results\predictions\pred_best_model
C:\Users\user\anaconda3\envs\tf_gpu\lib\site-packages\albumentations\__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.5 (you have 1.4.18). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
已設定中文字體: C:/Windows/Fonts/msjh.ttc
已啟用GPU記憶體增長
載入模型: results\unet_20250415_0656_20250415-065602\checkpoints\best_model.h5
2025-04-15 07:35:34.693871: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-04-15 07:35:35.088450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21344 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:01:00.0, compute capability: 8.9
模型載入成功
使用驗證集進行預測...
開始評估模型...
2025-04-15 07:35:36.232469: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
2025-04-15 07:35:37.350084: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
1/1 [==============================] - 2s 2s/step
1/1 [==============================] - 0s 15ms/step
1/1 [==============================] - 0s 14ms/step
1/1 [==============================] - 0s 15ms/step
1/1 [==============================] - 0s 14ms/step
...
1/1 [==============================] - 0s 16ms/step
像素準確率: 0.8748
Dice係數: 0.8749
IoU係數: 0.7777

圖像級別的分類性能:
準確率: 0.7745
敏感度 (召回率): 0.9880
特異度: 0.5610
精確度: 0.6924
混淆矩陣已保存至 results\predictions\pred_best_model\confusion_matrix.png
評估結果已保存至 results\predictions\pred_best_model\evaluation_results.json
1/1 [==============================] - 0s 14ms/step
預測程序執行完畢
錯誤預測完成! 結果保存在: results\predictions\pred_best_model

按Enter返回主選單...