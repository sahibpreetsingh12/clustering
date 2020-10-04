from data_gen import datasets

import numpy as np

from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt

# 1.choose the k number of clusters.    
# 2. select k random points from data as centroids.

arr=np.array([[1,2],[3,1],[2,2],[2,1]
            ,[7,7],[8,8],[9,9],[7,8]])

# for initialising centroids
def init_clusters(arr,n_clusters=2):
    index_c=np.random.randint(arr.shape[0],size=n_clusters)

    return arr[index_c]

# classes array to store class assigned to each point 
classes=np.zeros(shape=arr.shape[0])

# distances array it will of shape (n,k) where n is no  of data points and k is no of centroids




