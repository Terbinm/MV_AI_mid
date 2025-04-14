"""
配置參數檔案
包含整個專案的全域參數配置
"""

import os
import datetime


class Config:
    """配置類，包含所有專案參數"""

    def __init__(self):
        # 基本路徑
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.RESULT_DIR = os.path.join(self.BASE_DIR, 'results')

        # 確保目錄存在
        os.makedirs(self.RESULT_DIR, exist_ok=True)

        # 資料集配置
        self.CSV_FILE = 'caxton_dataset_filtered_no_outliers_img_info.csv'  # 主要資料集CSV檔案
        # self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'limited_data_size')
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, 'processed')

        # 資料處理參數
        self.MAX_SAMPLES = 100000  # 最大樣本數，None表示使用全部資料
        self.BALANCE_DATASET = True  # 是否平衡正負樣本
        self.BALANCE_RATIO = 0.5  # 錯誤樣本比例
        self.VAL_RATIO = 0.2  # 驗證集比例
        self.CROP_SIZE = 320  # 裁剪尺寸
        self.RESIZE = 224  # 調整大小

        # 模型參數
        self.MODEL_NAME = f"unet_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        self.INPUT_SIZE = (224, 224, 3)  # 模型輸入尺寸
        self.NUM_CLASSES = 1  # 二元分割
        self.BASE_FILTERS = 32  # 基礎卷積濾波器數量
        self.DEPTH = 4  # U-Net深度
        self.DROPOUT_RATE = 0.1  # Dropout比率
        self.USE_BATCHNORM = True  # 是否使用批次正規化

        # 訓練參數
        self.BATCH_SIZE = 16  # 批次大小
        self.EPOCHS = 50  # 訓練輪數
        self.LEARNING_RATE = 1e-4  # 學習率
        self.LOSS_TYPE = 'bce_dice'  # 損失函數類型，可選: 'bce', 'dice', 'bce_dice'
        self.EARLY_STOPPING_PATIENCE = 10  # 提前停止耐心值

        # 預測參數
        self.THRESHOLD = 0.5  # 二值化閾值

        # 硬體設定
        self.NUM_WORKERS = 4  # 資料載入工作執行緒數
        self.USE_MULTIPROCESSING = True  # 是否使用多處理
        self.GPU_MEMORY_GROWTH = True  # 是否啟用GPU記憶體增長

        # 日誌設定
        self.LOG_DIR = os.path.join(self.RESULT_DIR, 'logs')
        os.makedirs(self.LOG_DIR, exist_ok=True)

    def update_from_dict(self, config_dict):
        """從字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, filepath):
        """儲存配置到檔案"""
        with open(filepath, 'w') as f:
            for key, value in vars(self).items():
                if not key.startswith('__'):
                    f.write(f"{key} = {value}\n")

    @classmethod
    def load_from_file(cls, filepath):
        """從檔案載入配置"""
        config = cls()
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # 處理不同類型的值
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'none':
                            value = None
                        elif value.startswith('(') and value.endswith(')'):
                            # 元組處理
                            try:
                                value = eval(value)
                            except:
                                pass
                        elif value.replace('.', '', 1).isdigit():
                            # 數字處理
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except:
                                pass

                        if hasattr(config, key):
                            setattr(config, key, value)
        return config


if __name__ == "__main__":
    # 測試配置類
    config = Config()
    print("配置參數:")
    for key, value in vars(config).items():
        if not key.startswith('__'):
            print(f"{key} = {value}")