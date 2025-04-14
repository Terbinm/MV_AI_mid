import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import os


def configure_chinese_font():
    """設定matplotlib使用微軟正黑體"""
    font_path = 'C:/Windows/Fonts/msjh.ttc'  # 微軟正黑體

    if os.path.exists(font_path):
        plt.rcParams['font.family'] = 'sans-serif'
        fontP = FontProperties(fname=font_path)
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        print(f"已設定中文字體: {font_path}")
        return fontP
    else:
        print(f"警告: 找不到指定字體 {font_path}")
        print("嘗試使用系統預設字體")
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            print("設定系統預設字體失敗")
        return None


# 在import此模組時自動設定中文字體
chinese_font = configure_chinese_font()