為什麼需要data_generator.py?我有幾百萬的資料集大小,我甚至需要無視一部分資料!


'''
正在執行資料預處理...
是否清理先前處理的資料? [y/N]: y
正在刪除處理後的資料...
已成功刪除目錄: D:\led\MV_AI_mid\data\processed
開始資料預處理流程...
分析CSV資料: D:\led\MV_AI_mid\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv
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
D:\led\MV_AI_mid\utils\preprocess.py:131: UserWarning: Glyph 27491 (\N{CJK UNIFIED IDEOGRAPH-6B63}) missing from font(s) DejaVu Sans.
  plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
D:\led\MV_AI_mid\utils\preprocess.py:131: UserWarning: Glyph 24120 (\N{CJK UNIFIED IDEOGRAPH-5E38}) missing from font(s) DejaVu Sans.
  plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
D:\led\MV_AI_mid\utils\preprocess.py:131: UserWarning: Glyph 37679 (\N{CJK UNIFIED IDEOGRAPH-932F}) missing from font(s) DejaVu Sans.
  plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
D:\led\MV_AI_mid\utils\preprocess.py:131: UserWarning: Glyph 35492 (\N{CJK UNIFIED IDEOGRAPH-8AA4}) missing from font(s) DejaVu Sans.
  plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
C:\Users\user\AppData\Local\Programs\PyCharm Professional 2024.3.1.1\plugins\python-ce\helpers\pycharm_matplotlib_backend\backend_interagg.py:124: UserWarning: Glyph 27491 (\N{CJK UNIFIED IDEOGRAPH-6B63}) missing from font(s) DejaVu Sans.
  FigureCanvasAgg.draw(self)
C:\Users\user\AppData\Local\Programs\PyCharm Professional 2024.3.1.1\plugins\python-ce\helpers\pycharm_matplotlib_backend\backend_interagg.py:124: UserWarning: Glyph 24120 (\N{CJK UNIFIED IDEOGRAPH-5E38}) missing from font(s) DejaVu Sans.
  FigureCanvasAgg.draw(self)
C:\Users\user\AppData\Local\Programs\PyCharm Professional 2024.3.1.1\plugins\python-ce\helpers\pycharm_matplotlib_backend\backend_interagg.py:124: UserWarning: Glyph 37679 (\N{CJK UNIFIED IDEOGRAPH-932F}) missing from font(s) DejaVu Sans.
  FigureCanvasAgg.draw(self)
C:\Users\user\AppData\Local\Programs\PyCharm Professional 2024.3.1.1\plugins\python-ce\helpers\pycharm_matplotlib_backend\backend_interagg.py:124: UserWarning: Glyph 35492 (\N{CJK UNIFIED IDEOGRAPH-8AA4}) missing from font(s) DejaVu Sans.
  FigureCanvasAgg.draw(self)

錯誤類型分佈:
  : 156539 (100.00%)
檢查影像路徑: D:\led\MV_AI_mid\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv
檢查影像: 100%|██████████| 156539/156539 [00:04<00:00, 38812.39it/s]
檢查完成: 共 156539 個有效影像, 0 個缺失影像
抽樣並探索影像: D:\led\MV_AI_mid\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv
處理正常樣本...
處理錯誤樣本...
已保存 5 個正常樣本和 5 個錯誤樣本到 D:\led\MV_AI_mid\data\plt\sample_images
創建平衡樣本: D:\led\MV_AI_mid\data\limited_data_size\caxton_dataset_filtered_no_outliers_img_info.csv -> D:\led\MV_AI_mid\data\processed\balanced_dataset.csv
原始資料集: 156539 筆
原始分佈: 正常=1940, 錯誤=154599
平衡後分佈: 正常=1940, 錯誤=50000
平衡後總樣本數: 51940
平衡樣本已保存至: D:\led\MV_AI_mid\data\processed\balanced_dataset.csv
2025-04-14 20:54:01.672829: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-04-14 20:54:02.202282: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
載入資料集: D:\led\MV_AI_mid\data\processed\balanced_dataset.csv
載入完成，共 51940 筆資料
訓練集大小: 41552, 驗證集大小: 10388
索引檔案已保存至 D:\led\MV_AI_mid\data\processed\train_indices.csv 和 D:\led\MV_AI_mid\data\processed\val_indices.csv
預處理訓練集影像...
Corrupt JPEG data: 1 extraneous bytes before marker 0xd1
Corrupt JPEG data: 1 extraneous bytes before marker 0xd5
'''