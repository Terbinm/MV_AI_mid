# 測試不同的平衡策略
python toolbox/balance_sample.py --csv data/limited_data_size/caxton_dataset_filtered_no_outliers_img_info.csv --output data/balanced_test --mode test --max_samples 100000

# 創建用於模型訓練的平衡資料集
python toolbox/balance_sample.py --csv data/limited_data_size/caxton_dataset_filtered_no_outliers_img_info.csv --output data/processed --mode create --strategy hybrid --ratio 0.5 --max_samples 10000