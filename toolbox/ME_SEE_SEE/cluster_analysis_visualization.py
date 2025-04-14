import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# 設定路徑 & 取樣
csv_path = r"C:\led_code\MV_AI_mid_ssd\toolbox\ME_SEE_SEE\caxton_dataset_filtered_no_outliers_img_info.csv"
sample_size = 5000  # 根據電腦記憶體大小可調整
df = pd.read_csv(csv_path, usecols=["flow_rate", "feed_rate", "z_offset", "hotend"])
df_sample = df.sample(n=sample_size, random_state=42)

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)

# ---------------- PCA 降維 ----------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.title("PCA Projection (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# ---------------- t-SNE 降維 ----------------
tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1])
plt.title("t-SNE Projection (2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.show()

# ---------------- Dendrogram (階層式聚類) ----------------
linked = linkage(X_scaled, method='ward')  # 'ward' 最適合用於數值距離

plt.figure(figsize=(10, 5))
dendrogram(linked,
           truncate_mode='level',  # 只顯示部分分支
           p=10,  # 顯示10層
           leaf_rotation=90.,
           leaf_font_size=10.,
           show_contracted=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index (truncated)")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# ---------------- Agglomerative Clustering 分群 + 投影 ----------------
n_clusters = 3  # 你可以自由嘗試其他群數
cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = cluster.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2')
plt.title(f"PCA Clustering (k={n_clusters})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='Set2')
plt.title(f"t-SNE Clustering (k={n_clusters})")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
