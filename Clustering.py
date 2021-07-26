#%%
# Libraries for data loading
import lasio
import pandas as pd
import os

# Libraries for data visualization:
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# Read the Files
las_file_list = []
path = '/Users/boyun/Desktop/SMU/Clustering/'

# View contents of the path, we will see LAS files plus an ASCII file
files = os.listdir(path)
files

#%%
# Only the LAS files get added to an empty array
for file in files:
    if file.lower().endswith('.las'):
        las_file_list.append(path +'/' + file)
      
las_file_list


#%%
# Extracting only Depth and Slowstrain values and Creating a dataframe
df_list = []

for lasfile in las_file_list:
    las = lasio.read(lasfile,ignore_header_errors=True)
    lasdf = las.df()
    
    df_list.append(lasdf)


df_list[0]


#%%
npArray=df_list[0].to_numpy()

# %%
# K means clustering
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

pca=PCA(n_components=2)
pca_result=pca.fit_transform(npArray)
kmeans=KMeans(n_clusters=2).fit(pca_result)
kmean_group=kmeans.predict(pca_result)
centers=kmeans.cluster_centers_
labels=kmeans.labels_

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmean_group, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

