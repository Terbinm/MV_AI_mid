"""
主程式進入點
3D列印錯誤偵測系統主程式
"""

import os
import argparse
import sys
from config.config import Config
from utils.font_utils import chinese_font


def show_banner():
    """顯示程式橫幅"""
    banner = """
    =======================================================
      3D列印錯誤偵測系統 - 基於影像分割的自動錯誤檢測
    =======================================================
    """
    print(banner)


def display_menu():
    """顯示主選單"""
    menu = """
    請選擇操作:
    1. 資料預處理
    2. 模型訓練
    3. 模型評估
    4. 錯誤預測
    5. 觀察資料分布
    6. 測試資料平衡策略
    7. 退出
    """
    print(menu)


def process_data():
    """執行資料預處理"""
    print("正在執行資料預處理...")

    # 引入預處理模組
    from utils.preprocess import preprocess_all_data, cleanup_processed_data

    # 載入配置
    config = Config()

    # 詢問是否清理先前處理的資料
    response = input("是否清理先前處理的資料? [y/N]: ")
    if response.lower() == 'y':
        cleanup_processed_data(config, confirm=False)

    # 詢問是否自訂平衡比例
    customize_balance = input("是否自訂資料平衡參數? [y/N]: ")
    if customize_balance.lower() == 'y':
        # 設置平衡比例
        try:
            ratio = float(input(f"設置錯誤樣本比例 (0.0-1.0) [預設: {config.BALANCE_RATIO}]: "))
            if 0.0 <= ratio <= 1.0:
                config.BALANCE_RATIO = ratio
        except ValueError:
            print("使用預設值")

        # 設置平衡策略
        strategy = input(f"選擇平衡策略 (undersample/oversample/hybrid) [預設: {config.BALANCE_STRATEGY}]: ")
        if strategy in ['undersample', 'oversample', 'hybrid']:
            config.BALANCE_STRATEGY = strategy

        # 設置最大樣本數
        try:
            max_samples = int(input(f"設置最大樣本數 [預設: {config.MAX_SAMPLES}]: "))
            if max_samples > 0:
                config.MAX_SAMPLES = max_samples
        except ValueError:
            print("使用預設值")

    # 執行預處理
    preprocess_all_data(config)

    print("資料預處理完成!")


def train_model():
    """執行模型訓練"""
    print("正在執行模型訓練...")

    # 引入訓練模組
    from train import train_model as start_training

    # 載入配置
    config = Config()

    # 詢問是否自訂訓練參數
    customize_training = input("是否自訂訓練參數? [y/N]: ")
    if customize_training.lower() == 'y':
        # 設置批次大小
        try:
            batch_size = int(input(f"設置批次大小 [預設: {config.BATCH_SIZE}]: "))
            if batch_size > 0:
                config.BATCH_SIZE = batch_size
        except ValueError:
            print("使用預設值")

        # 設置訓練輪數
        try:
            epochs = int(input(f"設置訓練輪數 [預設: {config.EPOCHS}]: "))
            if epochs > 0:
                config.EPOCHS = epochs
        except ValueError:
            print("使用預設值")

        # 設置學習率
        try:
            lr = float(input(f"設置學習率 [預設: {config.LEARNING_RATE}]: "))
            if lr > 0:
                config.LEARNING_RATE = lr
        except ValueError:
            print("使用預設值")

        # 設置提前停止耐心值
        try:
            patience = int(input(f"設置提前停止耐心值 [預設: {config.EARLY_STOPPING_PATIENCE}]: "))
            if patience > 0:
                config.EARLY_STOPPING_PATIENCE = patience
        except ValueError:
            print("使用預設值")

    # 執行訓練
    try:
        start_training()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    print("模型訓練程序完成!")


def evaluate_model():
    """執行模型評估"""
    print("正在執行模型評估...")

    # 引入評估模組
    import evaluate

    # 列出可用模型
    models = list_available_models()
    if not models:
        print("沒有找到可用的模型，請先訓練模型")
        return

    # 選擇模型
    print("\n可用模型:")
    for i, model_path in enumerate(models):
        print(f"{i + 1}. {os.path.basename(model_path)}")

    try:
        choice = int(input("\n請選擇模型編號: "))
        if choice < 1 or choice > len(models):
            print("無效的選擇")
            return

        selected_model = models[choice - 1]
    except ValueError:
        print("請輸入有效的數字")
        return

    # 選擇評估模式
    print("\n評估模式:")
    print("1. 驗證集評估")
    print("2. 自定義CSV評估")

    try:
        mode_choice = int(input("\n請選擇評估模式: "))
        if mode_choice < 1 or mode_choice > 2:
            print("無效的選擇")
            return
    except ValueError:
        print("請輸入有效的數字")
        return

    # 載入配置
    config = Config()

    # 詢問是否自訂閾值
    customize_threshold = input("是否自訂二值化閾值? [y/N]: ")
    threshold_value = None
    if customize_threshold.lower() == 'y':
        try:
            threshold_value = float(input(f"設置閾值 (0.0-1.0) [預設: {config.THRESHOLD}]: "))
            if 0.0 <= threshold_value <= 1.0:
                print(f"使用自訂閾值: {threshold_value}")
            else:
                threshold_value = None
                print("閾值超出範圍，使用默認值")
        except ValueError:
            print("無效輸入，使用默認值")

    # 建立評估命令並執行
    cmd = ["python", "evaluate.py", "--model", selected_model]

    if mode_choice == 1:
        cmd.extend(["--mode", "validation"])
    else:
        cmd.extend(["--mode", "custom"])

        csv_path = input("請輸入CSV檔案路徑: ")
        if not os.path.exists(csv_path):
            print(f"找不到檔案: {csv_path}")
            return

        image_dir = input("請輸入影像目錄: ")
        if not os.path.exists(image_dir):
            print(f"找不到目錄: {image_dir}")
            return

        cmd.extend(["--csv", csv_path, "--image_dir", image_dir])

    output_dir = os.path.join("results", "evaluation", f"eval_{os.path.basename(selected_model).split('.')[0]}")
    cmd.extend(["--output", output_dir])

    # 加入閾值參數
    if threshold_value is not None:
        cmd.extend(["--threshold", str(threshold_value)])

    # 執行評估
    import subprocess
    print(f"\n執行命令: {' '.join(cmd)}")
    subprocess.run(cmd)

    print(f"模型評估完成! 結果保存在: {output_dir}")


def predict_errors():
    """執行錯誤預測"""
    print("正在執行錯誤預測...")

    # 引入預測模組
    import predict

    # 列出可用模型
    models = list_available_models()
    if not models:
        print("沒有找到可用的模型，請先訓練模型")
        return

    # 選擇模型
    print("\n可用模型:")
    for i, model_path in enumerate(models):
        print(f"{i + 1}. {os.path.basename(model_path)}")

    try:
        choice = int(input("\n請選擇模型編號: "))
        if choice < 1 or choice > len(models):
            print("無效的選擇")
            return

        selected_model = models[choice - 1]
    except ValueError:
        print("請輸入有效的數字")
        return

    # 選擇預測模式
    print("\n預測模式:")
    print("1. 驗證集預測")
    print("2. 目錄預測 (預測指定目錄中的所有影像)")
    print("3. CSV預測 (根據CSV檔案中的路徑預測)")

    try:
        mode_choice = int(input("\n請選擇預測模式: "))
        if mode_choice < 1 or mode_choice > 3:
            print("無效的選擇")
            return
    except ValueError:
        print("請輸入有效的數字")
        return

    # 載入配置
    config = Config()

    # 詢問是否自訂閾值
    customize_threshold = input("是否自訂二值化閾值? [y/N]: ")
    threshold_value = None
    if customize_threshold.lower() == 'y':
        try:
            threshold_value = float(input(f"設置閾值 (0.0-1.0) [預設: {config.THRESHOLD}]: "))
            if 0.0 <= threshold_value <= 1.0:
                print(f"使用自訂閾值: {threshold_value}")
            else:
                threshold_value = None
                print("閾值超出範圍，使用默認值")
        except ValueError:
            print("無效輸入，使用默認值")

    # 建立預測命令並執行
    cmd = ["python", "predict.py", "--model", selected_model]

    if mode_choice == 1:
        cmd.extend(["--mode", "validation"])
    elif mode_choice == 2:
        cmd.extend(["--mode", "directory"])

        input_dir = input("請輸入影像目錄: ")
        if not os.path.exists(input_dir):
            print(f"找不到目錄: {input_dir}")
            return

        cmd.extend(["--input", input_dir])
    else:  # mode_choice == 3
        cmd.extend(["--mode", "csv"])

        csv_path = input("請輸入CSV檔案路徑: ")
        if not os.path.exists(csv_path):
            print(f"找不到檔案: {csv_path}")
            return

        image_dir = input("請輸入影像根目錄: ")
        if not os.path.exists(image_dir):
            print(f"找不到目錄: {image_dir}")
            return

        cmd.extend(["--input", csv_path, "--image_dir", image_dir])

    output_dir = os.path.join("results", "predictions", f"pred_{os.path.basename(selected_model).split('.')[0]}")
    cmd.extend(["--output", output_dir])

    # 加入閾值參數
    if threshold_value is not None:
        cmd.extend(["--threshold", str(threshold_value)])

    # 詢問是否生成視覺化影片
    if mode_choice == 3:
        video_choice = input("生成視覺化影片? [Y/n]: ")
        if video_choice.lower() == 'n':
            cmd.append("--no_video")

    # 執行預測
    import subprocess
    print(f"\n執行命令: {' '.join(cmd)}")
    subprocess.run(cmd)

    print(f"錯誤預測完成! 結果保存在: {output_dir}")


def explore_data():
    """觀察資料分布"""
    print("正在分析資料分布...")

    # 引入預處理模組
    from utils.preprocess import analyze_csv_data, sample_and_explore_images

    # 載入配置
    config = Config()

    # 選擇CSV檔案
    csv_files = list_csv_files()
    if not csv_files:
        print("沒有找到CSV檔案")
        return

    print("\n可用CSV檔案:")
    for i, csv_path in enumerate(csv_files):
        print(f"{i + 1}. {os.path.basename(csv_path)}")

    try:
        choice = int(input("\n請選擇CSV檔案編號: "))
        if choice < 1 or choice > len(csv_files):
            print("無效的選擇")
            return

        selected_csv = csv_files[choice - 1]
    except ValueError:
        print("請輸入有效的數字")
        return

    # 分析CSV檔案
    output_dir = os.path.join(config.DATA_DIR, "plt", "data_exploration")
    os.makedirs(output_dir, exist_ok=True)

    # 分析CSV
    stats = analyze_csv_data(selected_csv, output_dir)

    # 抽樣並探索影像
    sample_choice = input("\n是否抽樣並探索影像? [y/N]: ")
    if sample_choice.lower() == 'y':
        num_samples = 5
        try:
            num_input = input(f"要抽樣的影像數量 [預設: {num_samples}]: ")
            if num_input:
                num_samples = int(num_input)
        except ValueError:
            print("使用預設值")

        sample_and_explore_images(
            csv_path=selected_csv,
            image_dir=config.DATA_DIR,
            output_dir=os.path.join(output_dir, "sample_images"),
            num_samples=num_samples
        )

    print(f"資料分析完成! 結果保存在: {output_dir}")


def test_balance_strategies():
    """測試不同的資料平衡策略"""
    print("正在測試資料平衡策略...")

    # 引入工具模組
    from toolbox.balance_sample import test_different_strategies

    # 載入配置
    config = Config()

    # 選擇CSV檔案
    csv_files = list_csv_files()
    if not csv_files:
        print("沒有找到CSV檔案")
        return

    print("\n可用CSV檔案:")
    for i, csv_path in enumerate(csv_files):
        print(f"{i + 1}. {os.path.basename(csv_path)}")

    try:
        choice = int(input("\n請選擇CSV檔案編號: "))
        if choice < 1 or choice > len(csv_files):
            print("無效的選擇")
            return

        selected_csv = csv_files[choice - 1]
    except ValueError:
        print("請輸入有效的數字")
        return

    # 設置輸出目錄
    output_dir = os.path.join(config.DATA_DIR, "balanced_test")
    os.makedirs(output_dir, exist_ok=True)

    # 設置最大樣本數
    max_samples = config.MAX_SAMPLES
    try:
        max_input = input(f"設置最大樣本數 [預設: {max_samples}]: ")
        if max_input:
            max_samples = int(max_input)
    except ValueError:
        print("使用預設值")

    # 執行測試
    test_different_strategies(
        csv_path=selected_csv,
        output_dir=output_dir,
        max_samples=max_samples
    )

    print(f"資料平衡策略測試完成! 結果保存在: {output_dir}")


def list_available_models():
    """列出可用的模型檔案"""
    result_dirs = []

    # 搜尋results目錄中的所有子目錄
    for root, dirs, files in os.walk("results"):
        for file in files:
            if file.endswith(".h5"):
                result_dirs.append(os.path.join(root, file))

    return result_dirs


def list_csv_files():
    """列出可用的CSV檔案"""
    csv_files = []

    # 載入配置
    config = Config()

    # 搜尋data目錄中的CSV檔案
    data_dirs = [
        os.path.join(config.DATA_DIR, "limited_data_size"),
        os.path.join(config.DATA_DIR, "raw"),
        config.DATA_DIR
    ]

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(data_dir, file))

    return csv_files


def main():
    """主函數"""
    # 檢查必要目錄是否存在
    required_dirs = ["config", "data", "models", "utils", "results"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # 顯示橫幅
    show_banner()

    # 解析命令列引數
    parser = argparse.ArgumentParser(description='3D列印錯誤偵測系統')
    parser.add_argument('--preprocess', action='store_true', help='執行資料預處理')
    parser.add_argument('--train', action='store_true', help='執行模型訓練')
    parser.add_argument('--evaluate', action='store_true', help='執行模型評估')
    parser.add_argument('--predict', action='store_true', help='執行錯誤預測')
    parser.add_argument('--explore', action='store_true', help='觀察資料分布')
    parser.add_argument('--balance', action='store_true', help='測試資料平衡策略')

    args = parser.parse_args()

    # 執行指定操作
    if args.preprocess:
        process_data()
        return
    elif args.train:
        train_model()
        return
    elif args.evaluate:
        evaluate_model()
        return
    elif args.predict:
        predict_errors()
        return
    elif args.explore:
        explore_data()
        return
    elif args.balance:
        test_balance_strategies()
        return

    # 如果沒有命令列引數，顯示互動式選單
    while True:
        display_menu()
        choice = input("請輸入選項編號: ")

        if choice == '1':
            process_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            predict_errors()
        elif choice == '5':
            explore_data()
        elif choice == '6':
            test_balance_strategies()
        elif choice == '7':
            print("感謝使用，再見!")
            sys.exit(0)
        else:
            print("無效的選擇，請重新輸入")

        input("\n按Enter返回主選單...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程式已中止")
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()