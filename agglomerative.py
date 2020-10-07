from data_gen import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

arr=np.array([[1,1],[2,2],[1,3],[2,1]
            ,[9,9],[10,10],[8,8],
            [15,15],[14,15],[15,14]])

sns.scatterplot(x=arr[:,0],y=arr[:,1],hue=arr.shape[1])

distance_arr=np.zeros(shape=(arr.shape[0],arr.shape[0]))

print(distance_arr)

