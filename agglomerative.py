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


i=0
while i < arr.shape[0]:
    point=arr[i]
    for pointer in range(arr.shape[0]):
        distance_arr[pointer,i]=agg_clustering.get_distance(cen=point,arr=arr[pointer],axis=0)
    i+=1
print(distance_arr)
