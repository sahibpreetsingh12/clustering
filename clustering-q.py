from data_gen import datasets

import numpy as np

from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt

# 1. choose the k number of clusters.    
# 2. select k random points from data as centroids.
# 3. Assign all the points to the closest cluster centroid
# 4. Recompute the centroids of newly formed clusters
# 5. Repeat 3 and 4

class kmeans:

    def __init__(self,n_clusters=2,arr=None,max_iter=None):
        self.n_clusters=n_clusters # STEP1
        self.arr=arr
        self.max_iter=max_iter

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
        max_iter=self.max_iter

        # classes array to store class assigned to each point i.e each row of DATABASE
        classes=np.zeros(shape=arr.shape[0])

        # distances array it will of shape (n,k) where n is no  of data points and k is no of centroids
        distances=np.zeros(shape=(arr.shape[0],n_clusters))

        # initialised centroids
        centroids= self.init_clusters() # STEP2

        print('Initially centroid was',centroids)
        for iter in range(max_iter):

            for i,cen in enumerate(centroids):
                distances[:,i]= self.get_distance(cen=cen,arr=arr)

            # Determine class membership of each point
            # by picking the closest centroid
            classes = np.argmin(distances, axis=1) # STEP3
                                                   # axis=1 row wise euclidean distance is claculated
                                                   # axis=0 column wise euclidean distance is calculated

            for c in range(n_clusters):
                centroids[c] = np.mean(arr[classes == c], axis=0) # STEP4
        print('Finally centroids converged to ',centroids)
        
        
        
k1=kmeans(2,datasets[0],200) # first arg= number of clusters for kmeans ,second arg=dataset third arg=number of iterations
k1.kmeans_calc()