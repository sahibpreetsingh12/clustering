from data_gen import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clustering import Clustering
from numpy.linalg import norm
arr=np.array([[1,1],[2,2],[1,3],[2,1]
            ,[9,9],[10,10],[8,8],
            [15,15],[14,15],[15,14]])

agg_clustering=Clustering()

distance_arr=np.zeros(shape=(arr.shape[0],arr.shape[0]))

# for i in range(distance_arr.shape[0]):
#     distance_arr[:,i]=get_distance()
# print(norm(arr[0,:]-arr[9,:]))
print(agg_clustering.get_distance(cen=arr[0,:],arr=arr[2,:],axis=0))