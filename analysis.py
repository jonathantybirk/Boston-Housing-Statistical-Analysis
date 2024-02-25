import numpy as np
np.set_printoptions(precision=3,suppress=True)
import pandas as pd
import sklearn.decomposition as dcom
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
from scipy.linalg import svd
from scipy.stats import shapiro
from scipy.stats import probplot

data = pd.read_csv("BostonHousing cleaned.csv",sep=",",header=0)
header = data.columns.tolist()
data = np.asarray(data)

y = data[:,-1]

M, N = data.shape

for i in range(N):
    mean = np.mean(data[:,i])
    std = np.std(data[:,i])
    
    data[:,i] = (data[:,i] - mean) / std

n = 2
pca = dcom.PCA(n_components=n)

# ## PLOTTING 2D
# principal_components = pca.fit_transform(data[:, :-1])

# plt.figure(figsize=(8, 6), num="2D Plot")
# scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=data[:, -1], cmap='viridis',alpha=0.7,s=9)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Boston Housing Dataset')
# plt.colorbar(scatter, label='Housing Price')
# plt.grid(True)

# ## Plotting the feature weights
# components = pca.components_
# fig, ax = plt.subplots(figsize=(8, 6), num="Feature Weights")
# x_pos = np.arange(len(header[:-1]))

# width = 0.3
# spacing = 0.05
# group_width = n * width + (n - 1) * spacing
# start_pos = x_pos - group_width / 2 + width / 2

# for i in range(n):
#     bar_position = start_pos + i * (width + spacing)
#     ax.bar(bar_position, components[i], width, label=f'Principal Component {i+1}')

# ax.set_xticks(x_pos)
# ax.set_xticklabels(header[:-1], rotation=45, ha="right")
# ax.set_ylabel('Feature Weight')
# ax.set_title(f'PCA Component Feature Importance')
# ax.legend()
# plt.tight_layout()


# ## Plotting explained variance
# rho = svd(data[:,:-1])[1]**2 / (svd(data[:,:-1])[1]**2).sum()

# threshold = 0.8

# plt.figure(figsize=(8, 6), num="Explained Variance")
# plt.plot(range(1, len(rho) + 1), rho, "x-")
# plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
# # plt.plot([1, len(rho)], [threshold, threshold], "k--")
# plt.title("Variance explained by principal components")
# plt.xlabel("Principal component")
# plt.ylabel("Variance explained")
# plt.legend(["Individual", "Cumulative", "Threshold"])
# plt.grid()

# ## PLOTTING 3D
# # plt.figure(figsize=(8, 6))
# # ax = plt.axes(projection='3d')
# # sc = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=data[:, -1], cmap='viridis',alpha=0.7,s=9)
# # plt.colorbar(sc, label='Housing Price')
# # plt.show()

# ## Plotting Correlation matrix
# cov_matrix = np.cov(data, rowvar=False)

# plt.figure(figsize=(8, 6),num="Covariance Heat Map")
# plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
# plt.colorbar()
# plt.xticks(ticks=np.arange(len(header)), labels=header, rotation=45)
# plt.yticks(ticks=np.arange(len(header)), labels=header)
# plt.title("Covariance Matrix Heatmap")
# plt.show()


# # Are attributes normally distributed?

# for i in range(len(header)):
#     stat, p = shapiro(data[:,i])
#     print(f'{header[i]}: Statistics={stat:.3f}, p={p:.3f}')

# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), num="Normalized feature distributions")
# axes = axes.flatten()

# for i in range(len(header), len(axes)):
#     fig.delaxes(axes[i])

# for i, column in enumerate(header):
#     axes[i].hist(data[:, i], bins=30, color='blue', alpha=0.7)
#     axes[i].set_title(column)

# plt.subplots_adjust(hspace=0.5)

# fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), num="Q-Q Plots")
# axes = axes.flatten()

# for i in range(len(header), len(axes)):
#     fig.delaxes(axes[i])

# for i, column in enumerate(header):
#     probplot(data[:, i], plot=axes[i])
#     axes[i].set_title(column)

# plt.subplots_adjust(hspace=0.5)




# Plot scatterplots of all attributes against medv
fig, axs = plt.subplots(4, 4, figsize=(12, 12), num="Scatterplots")
for j in range(13):
    axs[j//4, j%4].scatter(data[:, j], data[:, 13], c=y, cmap='viridis', alpha=0.7, s=9)
    axs[j//4, j%4].set_xlabel(header[j])
    axs[j//4, j%4].set_ylabel(header[13])
    axs[j//4, j%4].grid(True)

# Remove extra subplots
for j in range(13, 16):
    fig.delaxes(axs[j//4, j%4])

plt.tight_layout()
plt.show()