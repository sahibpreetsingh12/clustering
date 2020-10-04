from data_gen import datasets

import numpy as np

from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt

# 1.choose the k number of clusters.    
# 2. select k random points from data as centroids.

arr=np.array([[1,2],[3,1],[2,2],[2,1]
            ,[7,7],[8,8],[9,9],[7,8]])
class kmeans:

    def __init__(self,n_clusters=2,arr=arr):
        self.n_clusters=n_clusters
        self.arr=arr

    def get_distance(self,cen=None,arr=None):
        """Returns the distance the centroid is from each data point in points."""
        return np.linalg.norm(cen - arr, axis=1)
    
    # for initialising centroids
    def init_clusters(self):

        # if arr is None: #if we dont pass array the array used to initialise the object will be passed
        arr=self.arr
        n_clusters=self.n_clusters

        index_c=np.random.randint(arr.shape[0],size=n_clusters)

        return arr[index_c]

    def kmeans_calc(self,max_iter=50):

        #if we dont pass array the array used to initialise the object will be passed
        arr=self.arr
        n_clusters=self.n_clusters

        # classes array to store class assigned to each point 
        classes=np.zeros(shape=arr.shape[0])

        # distances array it will of shape (n,k) where n is no  of data points and k is no of centroids
        distances=np.zeros(shape=(arr.shape[0],n_clusters))

        # initialised centroids
        centroids= self.init_clusters()
        
        for iter in range(max_iter):

            for i,cen in enumerate(centroids):
                distances[:,i]= self.get_distance(cen=cen,arr=arr)
            # Determine class membership of each point
            # by picking the closest centroid
            classes = np.argmin(distances, axis=1)
        print(classes)
        print(distances)
        
k1=kmeans(2,arr)
k1.kmeans_calc()