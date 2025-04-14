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