import os
import pandas as pd
import glob
from datetime import datetime


def print_directory_tree(start_path, output_file, max_files=15):
    """列印目錄結構，每個資料夾最多顯示max_files個檔案，忽略.venv資料夾"""
    output_file.write(f"\n目錄結構 (最多顯示每個資料夾{max_files}個檔案):\n")
    print(f"\n目錄結構 (最多顯示每個資料夾{max_files}個檔案):")

    for root, dirs, files in os.walk(start_path):
        # 忽略.venv資料夾
        if '.venv' in dirs:
            dirs.remove('.venv')
        if '.git' in dirs:
            dirs.remove('.git')
        if '.idea' in dirs:
            dirs.remove('.idea')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        if 'toolbox' in dirs:
            dirs.remove('toolbox')

        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        dir_name = os.path.basename(root) or os.path.basename(start_path)

        tree_line = f"{indent}{dir_name}/"
        output_file.write(tree_line + "\n")
        print(tree_line)

        # 限制顯示的檔案數量
        file_count = 0
        for f in files:
            if file_count < max_files:
                file_line = f"{indent}    {f}"
                output_file.write(file_line + "\n")
                print(file_line)
                file_count += 1
            else:
                remaining = len(files) - max_files
                if remaining > 0:
                    remaining_line = f"{indent}    ... 還有 {remaining} 個檔案未顯示"
                    output_file.write(remaining_line + "\n")
                    print(remaining_line)
                break


def print_and_save_csv_content():
    """列印目錄結構及每個CSV檔案的前10行內容並儲存為文字檔"""
    # 定義路徑 - 使用絕對路徑
    root_dir = 'D:\\led\\MV_AI_mid'
    data_dir = os.path.join(root_dir, 'data', 'raw')

    # 建立輸出目錄
    output_dir = os.path.join(root_dir, 'toolbox')
    os.makedirs(output_dir, exist_ok=True)

    # 創建輸出檔案
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"csv_exploration_{timestamp}.txt")

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        # 記錄時間資訊
        time_info = f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        output_file.write(time_info + "\n\n")
        print(time_info)

        # 顯示目錄結構
        print_directory_tree(root_dir, output_file)

        # 尋找所有CSV檔案 - 使用絕對路徑
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

        # 顯示找到的CSV檔案數量
        csv_files_info = f"\n找到 {len(csv_files)} 個CSV檔案："
        output_file.write(csv_files_info + "\n")
        print(csv_files_info)
        for csv_file in csv_files:
            output_file.write(f"- {os.path.basename(csv_file)}\n")
            print(f"- {os.path.basename(csv_file)}")

        # 處理每個CSV檔案
        for csv_file in csv_files:
            try:
                # 從檔案路徑中提取檔案名稱
                file_name = os.path.basename(csv_file)
                separator = "=" * 80

                header_text = f"\n{separator}\n檔案: {file_name}\n{separator}"
                output_file.write(header_text + "\n")
                print(header_text)

                # 讀取CSV檔案
                df = pd.read_csv(csv_file)

                # 列印檔案資訊
                file_info = [
                    f"檔案大小: {os.path.getsize(csv_file) / 1024:.2f} KB",
                    f"資料筆數: {len(df)}",
                    f"欄位數量: {len(df.columns)}",
                    f"欄位名稱: {', '.join(df.columns)}"
                ]

                for info in file_info:
                    output_file.write(info + "\n")
                    print(info)

                # 列印前10行資料
                output_file.write("\n前10行資料:\n")
                print("\n前10行資料:")

                # 設定顯示選項
                pd.set_option('display.max_columns', None)  # 顯示所有欄位
                pd.set_option('display.width', 1000)  # 設定顯示寬度
                pd.set_option('display.max_rows', None)  # 確保顯示所有行

                # 使用to_string確保完整顯示
                data_sample = df.head(10).to_string(index=True)
                output_file.write(data_sample + "\n")
                print(data_sample)

            except Exception as e:
                error_message = f"處理檔案 {file_name} 時發生錯誤: {str(e)}"
                output_file.write(error_message + "\n")
                print(error_message)

        # 記錄完成資訊
        completion_message = f"\n執行完成，結果已儲存至 {output_filename}"
        output_file.write(completion_message)
        print(completion_message)


if __name__ == "__main__":
    print_and_save_csv_content()