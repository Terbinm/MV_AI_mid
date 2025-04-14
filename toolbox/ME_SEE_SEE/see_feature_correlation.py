import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# Load a sample of data to optimize performance
csv_path = r"C:\led_code\MV_AI_mid_ssd\toolbox\ME_SEE_SEE\caxton_dataset_filtered_no_outliers_img_info.csv"
sample_size = 10000  # You can adjust this based on available memory
df = pd.read_csv(csv_path, usecols=["flow_rate", "feed_rate", "z_offset", "hotend"])
df_sample = df.sample(n=sample_size, random_state=42)

# Standardize data before computing distances
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)

# Compute distance matrix and convert to square form
dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

# Compute correlation matrix for heatmap
corr_matrix = df_sample.corr()

# ----- Plot 1: Distance matrix (heatmap of pairwise distances -----
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix between Features")
plt.tight_layout()
plt.show()

# ----- Plot 2: KDE / Distribution -----
plt.figure(figsize=(10, 6))
for col in df_sample.columns:
    sns.kdeplot(df_sample[col], label=col, fill=True, alpha=0.4)
plt.title("Feature Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Save distance matrix as .npy for future use (no need to recompute)
# np.save("feature_distance_matrix.npy", dist_matrix)
