import numpy as np
np.set_printoptions(precision=3,suppress=True)
import pandas as pd
import sklearn.decomposition as dcom
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d

data = pd.read_csv("BostonHousing.csv",sep=",",header=0)
data = np.asarray(data)

M, N = data.shape

for i in range(N - 1):
    mean = np.mean(data[:,i])
    std = np.std(data[:,i])
    
    data[:,i] = (data[:,i] - mean) / std

pca = dcom.PCA(n_components=3)

principal_components = pca.fit_transform(data[:, :-1])

# PLOTTING 3D
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
sc = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=data[:, -1], cmap='viridis',alpha=0.7,s=9)
plt.colorbar(sc, label='Housing Price')
plt.show()

# # PLOTTING, 2D# plt.figure(figsize=(8, 6))
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=data[:, -1], cmap='viridis',alpha=0.7,s=9)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Boston Housing Dataset')
# plt.colorbar(scatter, label='Housing Price')
# plt.grid(True)
# plt.show()