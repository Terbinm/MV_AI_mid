"""
資料集平衡工具
用於測試不同的資料平衡策略，解決3D列印錯誤偵測中樣本不平衡問題
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# 修正導入路徑，使此腳本可以在toolbox目錄中執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.font_utils import configure_chinese_font, chinese_font


def analyze_label_distribution(df, error_label_col='error_label'):
    """
    分析資料集的標籤分布

    Args:
        df: 資料DataFrame
        error_label_col: 錯誤標籤列名

    Returns:
        (normal_count, error_count): 正常和錯誤樣本數量
    """
    # 確保錯誤標籤存在
    if error_label_col not in df.columns:
        print(f"創建'{error_label_col}'欄位...")
        df[error_label_col] = ((df['flow_rate_class'] != 1) |
                               (df['feed_rate_class'] != 1) |
                               (df['z_offset_class'] != 1) |
                               (df['hotend_class'] != 1)).astype(int)

    # 計算分布
    normal_count = (df[error_label_col] == 0).sum()
    error_count = (df[error_label_col] == 1).sum()

    print(f"標籤分布:")
    print(f"  - 正常樣本: {normal_count} ({normal_count / len(df) * 100:.2f}%)")
    print(f"  - 錯誤樣本: {error_count} ({error_count / len(df) * 100:.2f}%)")

    return normal_count, error_count


def balance_dataset_by_sampling(df, balance_ratio=0.5, max_samples=None, strategy='undersample', random_state=42):
    """
    通過抽樣平衡資料集

    Args:
        df: 資料DataFrame
        balance_ratio: 錯誤樣本佔總樣本的比例 (0.0-1.0)
        max_samples: 最大樣本數量，None表示不限制
        strategy: 平衡策略 ('undersample', 'oversample', 'hybrid')
        random_state: 隨機種子

    Returns:
        平衡後的DataFrame
    """
    # 確保錯誤標籤存在
    if 'error_label' not in df.columns:
        df['error_label'] = ((df['flow_rate_class'] != 1) |
                             (df['feed_rate_class'] != 1) |
                             (df['z_offset_class'] != 1) |
                             (df['hotend_class'] != 1)).astype(int)

    # 分離正常和錯誤樣本
    normal_df = df[df['error_label'] == 0]
    error_df = df[df['error_label'] == 1]

    normal_count = len(normal_df)
    error_count = len(error_df)

    print(f"原始分布:")
    print(f"  - 正常樣本: {normal_count} ({normal_count / len(df) * 100:.2f}%)")
    print(f"  - 錯誤樣本: {error_count} ({error_count / len(df) * 100:.2f}%)")

    # 設置隨機種子
    np.random.seed(random_state)

    # 計算目標樣本數
    if max_samples is None:
        max_samples = len(df)

    if strategy == 'undersample':
        # 下採樣策略：基於少數類別(正常樣本)的數量，對多數類別進行下採樣
        target_error_count = int(normal_count / (1 - balance_ratio) * balance_ratio)
        target_error_count = min(target_error_count, error_count)
        target_normal_count = normal_count

        # 最大樣本數限制
        if target_normal_count + target_error_count > max_samples:
            ratio = max_samples / (target_normal_count + target_error_count)
            target_normal_count = int(target_normal_count * ratio)
            target_error_count = int(target_error_count * ratio)

        # 抽樣
        if target_normal_count < normal_count:
            sampled_normal = normal_df.sample(target_normal_count)
        else:
            sampled_normal = normal_df

        sampled_error = error_df.sample(target_error_count)

    elif strategy == 'oversample':
        # 過採樣策略：基於多數類別(錯誤樣本)的數量，對少數類別進行過採樣
        target_normal_count = int(error_count / balance_ratio * (1 - balance_ratio))
        target_error_count = error_count

        # 最大樣本數限制
        if target_normal_count + target_error_count > max_samples:
            ratio = max_samples / (target_normal_count + target_error_count)
            target_normal_count = int(target_normal_count * ratio)
            target_error_count = int(target_error_count * ratio)

        # 抽樣 - 對少數類進行放回抽樣
        if target_normal_count > normal_count:
            # 過採樣：放回抽樣
            indices = np.random.choice(normal_df.index, size=target_normal_count, replace=True)
            sampled_normal = normal_df.loc[indices].reset_index(drop=True)
        else:
            sampled_normal = normal_df.sample(target_normal_count)

        sampled_error = error_df.sample(target_error_count)

    elif strategy == 'hybrid':
        # 混合策略：同時下採樣多數類和過採樣少數類
        # 計算理想的平衡後總樣本數
        if max_samples is None:
            # 預設使用現有樣本數的一半
            target_total = len(df) // 2
        else:
            target_total = max_samples

        # 計算平衡後的錯誤和正常樣本數
        target_error_count = int(target_total * balance_ratio)
        target_normal_count = target_total - target_error_count

        # 抽樣
        if target_normal_count <= normal_count:
            # 正常樣本不需要過採樣
            sampled_normal = normal_df.sample(target_normal_count)
        else:
            # 正常樣本需要過採樣
            indices = np.random.choice(normal_df.index, size=target_normal_count, replace=True)
            sampled_normal = normal_df.loc[indices].reset_index(drop=True)

        if target_error_count <= error_count:
            # 錯誤樣本不需要過採樣
            sampled_error = error_df.sample(target_error_count)
        else:
            # 錯誤樣本需要過採樣
            indices = np.random.choice(error_df.index, size=target_error_count, replace=True)
            sampled_error = error_df.loc[indices].reset_index(drop=True)

    else:
        raise ValueError(f"不支援的平衡策略: {strategy}")

    # 合併並打亂順序
    balanced_df = pd.concat([sampled_normal, sampled_error])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 分析平衡後的分布
    normal_count = (balanced_df['error_label'] == 0).sum()
    error_count = (balanced_df['error_label'] == 1).sum()

    print(f"平衡後分布 (策略: {strategy}):")
    print(f"  - 正常樣本: {normal_count} ({normal_count / len(balanced_df) * 100:.2f}%)")
    print(f"  - 錯誤樣本: {error_count} ({error_count / len(balanced_df) * 100:.2f}%)")
    print(f"  - 總樣本數: {len(balanced_df)}")

    return balanced_df


def test_different_strategies(csv_path, output_dir, test_strategies=None, balance_ratios=None, max_samples=None):
    """
    測試不同的平衡策略並視覺化結果

    Args:
        csv_path: CSV檔案路徑
        output_dir: 輸出目錄
        test_strategies: 要測試的策略列表
        balance_ratios: 要測試的平衡比例列表
        max_samples: 最大樣本數限制
    """
    # 預設值
    if test_strategies is None:
        test_strategies = ['undersample', 'oversample', 'hybrid']

    if balance_ratios is None:
        balance_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]

    # 讀取CSV
    print(f"讀取資料: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"原始資料集大小: {len(df)}筆")

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 分析原始分布
    print("分析原始資料分布...")
    normal_count, error_count = analyze_label_distribution(df)

    # 儲存結果
    results = []

    # 測試不同策略和平衡比例
    for strategy in test_strategies:
        for ratio in balance_ratios:
            print(f"\n測試策略: {strategy}, 平衡比例: {ratio}")

            # 平衡資料集
            balanced_df = balance_dataset_by_sampling(
                df=df,
                balance_ratio=ratio,
                max_samples=max_samples,
                strategy=strategy
            )

            # 記錄結果
            normal_count = (balanced_df['error_label'] == 0).sum()
            error_count = (balanced_df['error_label'] == 1).sum()
            total_count = len(balanced_df)

            results.append({
                'strategy': strategy,
                'balance_ratio': ratio,
                'normal_count': normal_count,
                'error_count': error_count,
                'total_count': total_count,
                'normal_percent': normal_count / total_count * 100,
                'error_percent': error_count / total_count * 100
            })

            # 儲存平衡後的資料集
            output_file = os.path.join(output_dir, f"balanced_{strategy}_ratio{ratio}.csv")
            balanced_df.to_csv(output_file, index=False)
            print(f"已儲存平衡後的資料集: {output_file}")

            # 測試訓練/驗證集分割
            X_train, X_val, y_train, y_val = train_test_split(
                balanced_df.drop('error_label', axis=1),
                balanced_df['error_label'],
                test_size=0.2,
                stratify=balanced_df['error_label'],
                random_state=42
            )

            train_normal = (y_train == 0).sum()
            train_error = (y_train == 1).sum()
            val_normal = (y_val == 0).sum()
            val_error = (y_val == 1).sum()

            print(f"訓練集分布: 正常={train_normal} ({train_normal / len(y_train) * 100:.2f}%), " +
                  f"錯誤={train_error} ({train_error / len(y_train) * 100:.2f}%)")
            print(f"驗證集分布: 正常={val_normal} ({val_normal / len(y_val) * 100:.2f}%), " +
                  f"錯誤={val_error} ({val_error / len(y_val) * 100:.2f}%)")

    # 將結果轉換為DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "sampling_results.csv"), index=False)

    # 繪製結果圖表
    plot_sampling_results(results_df, output_dir)


def plot_sampling_results(results_df, output_dir):
    """
    繪製採樣結果圖表

    Args:
        results_df: 結果DataFrame
        output_dir: 輸出目錄
    """
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 繪製樣本數量圖表
    plt.figure(figsize=(15, 10))

    # 對每個策略分組
    strategies = results_df['strategy'].unique()

    # 設置顏色和樣式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # 繪製總樣本數量
    plt.subplot(2, 2, 1)
    for i, strategy in enumerate(strategies):
        strategy_df = results_df[results_df['strategy'] == strategy]
        plt.plot(strategy_df['balance_ratio'], strategy_df['total_count'],
                 label=strategy, color=colors[i], marker=markers[i], linewidth=2)

    plt.title('不同策略和平衡比例下的總樣本數')
    plt.xlabel('錯誤樣本比例')
    plt.ylabel('總樣本數')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 繪製正常樣本數量
    plt.subplot(2, 2, 2)
    for i, strategy in enumerate(strategies):
        strategy_df = results_df[results_df['strategy'] == strategy]
        plt.plot(strategy_df['balance_ratio'], strategy_df['normal_count'],
                 label=strategy, color=colors[i], marker=markers[i], linewidth=2)

    plt.title('不同策略和平衡比例下的正常樣本數')
    plt.xlabel('錯誤樣本比例')
    plt.ylabel('正常樣本數')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 繪製錯誤樣本數量
    plt.subplot(2, 2, 3)
    for i, strategy in enumerate(strategies):
        strategy_df = results_df[results_df['strategy'] == strategy]
        plt.plot(strategy_df['balance_ratio'], strategy_df['error_count'],
                 label=strategy, color=colors[i], marker=markers[i], linewidth=2)

    plt.title('不同策略和平衡比例下的錯誤樣本數')
    plt.xlabel('錯誤樣本比例')
    plt.ylabel('錯誤樣本數')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 繪製正常樣本百分比
    plt.subplot(2, 2, 4)
    for i, strategy in enumerate(strategies):
        strategy_df = results_df[results_df['strategy'] == strategy]
        plt.plot(strategy_df['balance_ratio'], strategy_df['normal_percent'],
                 label=strategy, color=colors[i], marker=markers[i], linewidth=2)

    # 添加理想線 (1 - balance_ratio)
    balance_ratios = results_df['balance_ratio'].unique()
    plt.plot(balance_ratios, [100 * (1 - r) for r in balance_ratios],
             '--', color='gray', label='理想值', linewidth=1)

    plt.title('不同策略和平衡比例下的正常樣本百分比')
    plt.xlabel('目標錯誤樣本比例')
    plt.ylabel('實際正常樣本百分比')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampling_strategies_comparison.png"), dpi=300)
    plt.close()

    # 繪製堆疊柱狀圖
    plt.figure(figsize=(15, 8))

    # 對每個策略和比例組合繪製一個柱狀圖
    bar_width = 0.25
    index = np.arange(len(balance_ratios))

    for i, strategy in enumerate(strategies):
        strategy_df = results_df[results_df['strategy'] == strategy].sort_values('balance_ratio')
        normal_values = strategy_df['normal_count'].values
        error_values = strategy_df['error_count'].values

        plt.bar(index + i * bar_width, normal_values, bar_width, label=f'{strategy} - 正常',
                color=colors[i], alpha=0.7)
        plt.bar(index + i * bar_width, error_values, bar_width, bottom=normal_values,
                label=f'{strategy} - 錯誤', color=colors[i], alpha=0.3, hatch='//')

    plt.xlabel('目標錯誤樣本比例')
    plt.ylabel('樣本數')
    plt.title('不同策略和平衡比例下的樣本數分布')
    plt.xticks(index + bar_width, [f"{r:.1f}" for r in balance_ratios])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampling_strategies_stacked.png"), dpi=300)
    plt.close()

    print(f"圖表已儲存至: {output_dir}")


def create_balanced_sets_for_model(csv_path, output_dir, strategy='hybrid', balance_ratio=0.5,
                                   max_samples=100000, train_val_ratio=0.2):
    """
    創建用於模型訓練的平衡資料集

    Args:
        csv_path: CSV檔案路徑
        output_dir: 輸出目錄
        strategy: 平衡策略
        balance_ratio: 錯誤樣本比例
        max_samples: 最大樣本數
        train_val_ratio: 驗證集比例
    """
    # 讀取CSV
    print(f"讀取資料: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"原始資料集大小: {len(df)}筆")

    # 確保錯誤標籤存在
    if 'error_label' not in df.columns:
        df['error_label'] = ((df['flow_rate_class'] != 1) |
                             (df['feed_rate_class'] != 1) |
                             (df['z_offset_class'] != 1) |
                             (df['hotend_class'] != 1)).astype(int)

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 平衡資料集
    print(f"使用策略 '{strategy}' 平衡資料集, 錯誤樣本比例: {balance_ratio}, 最大樣本數: {max_samples}")
    balanced_df = balance_dataset_by_sampling(
        df=df,
        balance_ratio=balance_ratio,
        max_samples=max_samples,
        strategy=strategy
    )

    # 將平衡後的資料集分割為訓練集和驗證集
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=train_val_ratio,
        stratify=balanced_df['error_label'],
        random_state=42
    )

    # 儲存訓練集和驗證集
    train_file = os.path.join(output_dir, "train_balanced.csv")
    val_file = os.path.join(output_dir, "val_balanced.csv")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    # 儲存訓練集和驗證集索引
    train_indices_file = os.path.join(output_dir, "train_indices.csv")
    val_indices_file = os.path.join(output_dir, "val_indices.csv")

    pd.DataFrame({'index': train_df.index}).to_csv(train_indices_file, index=False)
    pd.DataFrame({'index': val_df.index}).to_csv(val_indices_file, index=False)

    # 分析最終資料集
    print("\n訓練集分布:")
    analyze_label_distribution(train_df)

    print("\n驗證集分布:")
    analyze_label_distribution(val_df)

    print(f"\n已儲存訓練集和驗證集:")
    print(f"  - 訓練集: {train_file} ({len(train_df)}筆)")
    print(f"  - 驗證集: {val_file} ({len(val_df)}筆)")
    print(f"  - 訓練集索引: {train_indices_file}")
    print(f"  - 驗證集索引: {val_indices_file}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='3D列印錯誤偵測資料平衡工具')
    parser.add_argument('--csv', type=str, required=True, help='CSV檔案路徑')
    parser.add_argument('--output', type=str, default='./data/balanced', help='輸出目錄')
    parser.add_argument('--mode', type=str, choices=['test', 'create'], default='test',
                        help='模式: test (測試不同策略), create (創建平衡資料集)')
    parser.add_argument('--strategy', type=str, choices=['undersample', 'oversample', 'hybrid'],
                        default='hybrid', help='平衡策略')
    parser.add_argument('--ratio', type=float, default=0.5, help='錯誤樣本目標比例 (0.0-1.0)')
    parser.add_argument('--max_samples', type=int, default=100000, help='最大樣本數量')

    args = parser.parse_args()

    # 模式選擇
    if args.mode == 'test':
        # 測試不同策略
        test_different_strategies(
            csv_path=args.csv,
            output_dir=args.output,
            max_samples=args.max_samples
        )
    else:
        # 創建平衡資料集
        create_balanced_sets_for_model(
            csv_path=args.csv,
            output_dir=args.output,
            strategy=args.strategy,
            balance_ratio=args.ratio,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    # 配置中文字體
    chinese_font = configure_chinese_font()
    main()