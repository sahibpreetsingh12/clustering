from data_gen import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from clustering import Clustering # own clustering package
from numpy.linalg import norm

arr=np.array([[1,1],[2,2],[1,3],[2,1]
            ,[9,9],[10,10],[8,8],
            [15,15],[14,15],[15,14]])

agg_clustering=Clustering() #object of our package

distance_arr=np.zeros(shape=(arr.shape[0],arr.shape[0]))

infinite = math.inf #creating a infinite variable

# STEP 2 - creating proximity matrix
i=0
while i < arr.shape[0]:
    point=arr[i]
    for pointer in range(arr.shape[0]):
        if i > pointer: # creating a upper traingular matrix because distance of say [1,3] to [3,1] will be equal to distance 
            #bw [3,1] and [1,3] 
            distance_temp=agg_clustering.get_distance(cen=point,arr=arr[pointer],axis=0)
            if distance_temp!=0: # if distance of a point from another point is not equal to zero than it's ok
                distance_arr[pointer,i]= distance_temp

            else:# if distance is zero fill it with infinite because in next step we have to take argmin and zero will always be minimum
                # as we are taking euclidean distances so they can never be negative they can only be zero
                # so that's why we replaced zero with infinite
                distance_arr[pointer,i]=infinite
        else:
            distance_arr[pointer,i]=infinite


    i+=1




min_=np.where(distance_arr == np.amin(distance_arr))

list_of_cordinates=list(zip(min_[0],min_[1]))
for cord in list_of_cordinates:
    print(cord)
