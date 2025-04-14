"""
資料集處理類別
用於處理大型3D列印錯誤偵測資料集
"""

import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import Sequence
import albumentations as A


class DatasetProcessor:
    """
    資料集處理器: 負責從大型CSV中讀取資料並處理
    """

    def __init__(self, csv_path, image_dir, config):
        """
        初始化資料集處理器

        Args:
            csv_path: CSV文件路徑
            image_dir: 原始影像目錄
            config: 配置對象
        """
        self.config = config
        self.image_dir = image_dir

        # 讀取選定的CSV列數
        self.df = None
        self._load_csv(csv_path, config.MAX_SAMPLES)

    def _load_csv(self, csv_path, max_samples=None):
        """
        載入CSV檔案，可選擇只載入部分資料

        Args:
            csv_path: CSV文件路徑
            max_samples: 最大載入樣本數，None表示載入全部
        """
        print(f"載入資料集: {csv_path}")

        # 使用pandas讀取檔案
        if max_samples:
            # 只讀取指定行數
            self.df = pd.read_csv(csv_path, nrows=max_samples)
        else:
            self.df = pd.read_csv(csv_path)

        print(f"載入完成，共 {len(self.df)} 筆資料")

    def create_error_labels(self):
        """
        根據參數類別建立錯誤標籤
        標註邏輯: 任一參數類別不等於1(正常)時，判定為錯誤

        Returns:
            DataFrame: 添加錯誤標籤後的DataFrame
        """
        # 確保所需列存在
        required_cols = ['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"CSV必須包含以下列: {required_cols}")

        # 創建錯誤標籤列
        # 1 代表正常，其他值代表異常，所以檢查是否有任何參數不等於1
        self.df['error_label'] = ((self.df['flow_rate_class'] != 1) |
                                  (self.df['feed_rate_class'] != 1) |
                                  (self.df['z_offset_class'] != 1) |
                                  (self.df['hotend_class'] != 1)).astype(int)

        # 統計錯誤標籤分布
        error_count = self.df['error_label'].sum()
        normal_count = len(self.df) - error_count
        print(f"標籤統計: 正常={normal_count}, 錯誤={error_count}, 錯誤率={error_count / len(self.df) * 100:.2f}%")

        return self.df

    def split_train_val(self, val_ratio=0.2, random_state=42):
        """
        分割訓練集和驗證集

        Args:
            val_ratio: 驗證集比例
            random_state: 隨機種子

        Returns:
            train_indices, val_indices: 訓練集和驗證集的索引
        """
        # 創建錯誤標籤（如果尚未創建）
        if 'error_label' not in self.df.columns:
            self.create_error_labels()

        # 分層抽樣，確保錯誤/正常樣本在訓練和驗證集中的比例一致
        from sklearn.model_selection import train_test_split

        train_indices, val_indices = train_test_split(
            self.df.index,
            test_size=val_ratio,
            random_state=random_state,
            stratify=self.df['error_label']
        )

        print(f"訓練集大小: {len(train_indices)}, 驗證集大小: {len(val_indices)}")

        # 保存索引檔案
        os.makedirs(os.path.join(self.config.DATA_DIR, 'processed'), exist_ok=True)

        train_indices_df = pd.DataFrame({'index': train_indices})
        val_indices_df = pd.DataFrame({'index': val_indices})

        train_path = os.path.join(self.config.DATA_DIR, 'processed', 'train_indices.csv')
        val_path = os.path.join(self.config.DATA_DIR, 'processed', 'val_indices.csv')

        train_indices_df.to_csv(train_path, index=False)
        val_indices_df.to_csv(val_path, index=False)

        print(f"索引檔案已保存至 {train_path} 和 {val_path}")

        return train_indices, val_indices

    def preprocess_and_save_images(self, indices, target_dir, crop_size=320, resize=224):
        """
        預處理並保存影像及其遮罩

        Args:
            indices: 要處理的資料索引
            target_dir: 目標儲存目錄
            crop_size: 裁剪尺寸
            resize: 調整大小
        """
        # 確保目標目錄存在
        images_dir = os.path.join(target_dir, 'images')
        masks_dir = os.path.join(target_dir, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # 處理每一張影像
        for idx in indices:
            row = self.df.loc[idx]

            # 讀取原始影像
            img_path = os.path.join(self.image_dir, row['img_path'])
            if not os.path.isfile(img_path):
                print(f"警告: 無法找到影像 {img_path}")
                continue

            try:
                # 讀取並裁剪影像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 無法讀取影像 {img_path}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 使用噴嘴尖端座標裁剪影像
                if 'nozzle_tip_x' in row and 'nozzle_tip_y' in row:
                    nozzle_x = row['nozzle_tip_x']
                    nozzle_y = row['nozzle_tip_y']

                    # 以噴嘴為中心裁剪
                    x1 = max(0, nozzle_x - crop_size // 2)
                    y1 = max(0, nozzle_y - crop_size // 2)
                    x2 = min(img.shape[1], x1 + crop_size)
                    y2 = min(img.shape[0], y1 + crop_size)

                    # 如果裁剪區域超出影像邊界，調整起始點
                    if x2 - x1 < crop_size:
                        x1 = max(0, x2 - crop_size)
                    if y2 - y1 < crop_size:
                        y1 = max(0, y2 - crop_size)

                    img = img[y1:y2, x1:x2]

                # 調整大小
                img = cv2.resize(img, (resize, resize))

                # 生成遮罩 (如果錯誤標籤為1，整張遮罩都是1；否則全0)
                if row['error_label'] == 1:
                    mask = np.ones((resize, resize), dtype=np.uint8) * 255
                else:
                    mask = np.zeros((resize, resize), dtype=np.uint8)

                # 保存處理後的影像和遮罩
                out_img_path = os.path.join(images_dir, f"{idx}.png")
                out_mask_path = os.path.join(masks_dir, f"{idx}.png")

                cv2.imwrite(out_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(out_mask_path, mask)

            except Exception as e:
                print(f"處理影像 {img_path} 時發生錯誤: {e}")


class PrintDataGenerator(Sequence):
    """
    3D列印資料產生器
    用於高效率地批次載入資料並進行即時資料增強
    """

    def __init__(self, image_dir, indices, batch_size=8, is_training=True, input_size=(224, 224), normalize=True):
        """
        初始化資料產生器

        Args:
            image_dir: 處理後影像目錄
            indices: 使用的索引列表
            batch_size: 批次大小
            is_training: 是否用於訓練（訓練時會進行資料增強）
            input_size: 輸入尺寸
            normalize: 是否正規化像素值
        """
        self.image_dir = os.path.join(image_dir, 'images')
        self.mask_dir = os.path.join(image_dir, 'masks')
        self.indices = indices
        self.batch_size = batch_size
        self.is_training = is_training
        self.input_size = input_size
        self.normalize = normalize

        # 設定資料增強轉換
        if is_training:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            ])
        else:
            self.transform = None

    def __len__(self):
        """返回批次數量"""
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        """獲取一個批次的資料"""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_masks = []

        for index in batch_indices:
            # 讀取影像和遮罩
            img_path = os.path.join(self.image_dir, f"{index}.png")
            mask_path = os.path.join(self.mask_dir, f"{index}.png")

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0  # 正規化遮罩到 [0, 1]

            # 資料增強
            if self.is_training and self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

            # 調整大小
            img = cv2.resize(img, self.input_size)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)

            # 正規化影像
            if self.normalize:
                img = img / 255.0

            batch_images.append(img)
            batch_masks.append(mask)

        # 轉換為NumPy陣列
        X = np.array(batch_images)
        y = np.array(batch_masks)[..., np.newaxis]  # 添加通道維度

        return X, y

    def on_epoch_end(self):
        """每個訓練週期結束時打亂索引"""
        if self.is_training:
            np.random.shuffle(self.indices)


def load_dataset_from_indices(processed_dir, indices_file, batch_size=8, is_training=True, input_size=(224, 224)):
    """
    從索引文件載入資料集

    Args:
        processed_dir: 預處理資料目錄
        indices_file: 索引文件路徑
        batch_size: 批次大小
        is_training: 是否用於訓練
        input_size: 輸入尺寸

    Returns:
        data_generator: 資料產生器
    """
    # 讀取索引
    indices_df = pd.read_csv(indices_file)
    indices = indices_df['index'].values

    # 創建資料產生器
    generator = PrintDataGenerator(
        image_dir=processed_dir,
        indices=indices,
        batch_size=batch_size,
        is_training=is_training,
        input_size=input_size
    )

    return generator