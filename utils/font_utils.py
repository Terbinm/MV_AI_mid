import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import os
import platform


def configure_chinese_font():
    """設定matplotlib使用中文字體，根據不同作業系統尋找適合的字體"""
    system = platform.system()
    font_path = None

    if system == 'Windows':
        font_candidates = [
            'C:/Windows/Fonts/msjh.ttc',  # 微軟正黑體
            'C:/Windows/Fonts/mingliu.ttc',  # 細明體
            'C:/Windows/Fonts/simsun.ttc',  # 宋體
            'C:/Windows/Fonts/msyh.ttc'  # 微軟雅黑
        ]
        for font in font_candidates:
            if os.path.exists(font):
                font_path = font
                break
    elif system == 'Darwin':  # macOS
        font_candidates = [
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf'
        ]
        for font in font_candidates:
            if os.path.exists(font):
                font_path = font
                break
    elif system == 'Linux':
        font_candidates = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc'
        ]
        for font in font_candidates:
            if os.path.exists(font):
                font_path = font
                break

    if font_path:
        plt.rcParams['font.family'] = 'sans-serif'
        fontP = FontProperties(fname=font_path)
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        print(f"已設定中文字體: {font_path}")
        return fontP
    else:
        print("警告: 找不到適合的中文字體")
        print("嘗試使用系統預設字體")
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei',
                                               'WenQuanYi Micro Hei', 'AR PL UMing CN']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            print("設定系統預設字體失敗")
        return None


# 在import此模組時自動設定中文字體
chinese_font = configure_chinese_font()