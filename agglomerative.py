from data_gen import datasets # part of our own package
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from clustering import Clustering # own clustering package
from numpy.linalg import norm


linkage_dictionary={}
arr=np.array([[1,1],[2,2],[1,3],[2,1]
            ,[9,9],[10,10],[8,8],
            [15,15],[14,15],[15,14]])

agg_clustering=Clustering() #object of our package

infinite = math.inf #creating a infinite variable

# STEP 2 - creating proximity matrix
def proximity_matrix_gen(arr):
    distance_arr=np.zeros(shape=(arr.shape[0],arr.shape[0]))
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
    return distance_arr



def find_minimum_proximity(arr): # to find minimum of a array

    minimum_distance=np.amin(arr) # minimum distance bw 2 clusters

    min_=np.where(arr == np.amin(arr))

    list_of_cordinates=list(zip(min_[0],min_[1])) # finding list of coordinates that have minimum values in proximity matrix
    return list_of_cordinates,minimum_distance

proximity=proximity_matrix_gen(arr) # arr is our Dataset return = proximity matrix

list_of_coordinates,minimum_distance= find_minimum_proximity(proximity) # Input = Proximity matrix 
                                    #return = 1. minimum of proximity matrix and 2.coordinates that corresponds to minimum


new_array=np.empty((0,arr.shape[1]))  # empty array of shape 0,shape of arr[1] where arr is our dataset

# list_of_coordinates responds to those pair of points that have least euclidean distances in proximity matrix
# list_of_coordinates[0] responds to just first pair that had least euclidean distance
# because initially we had k clusters after step2 we'll have k-1 clusters although there can be multiple points that 
# form clusters because they have also least euclidean distance

linkage_matrix=np.empty((0,4)) #final linkage matrix

clustered_p_1=list_of_coordinates[0][0] 
clustered_p_2=list_of_coordinates[0][1]
cluster_1=[clustered_p_1,clustered_p_2]

print("cluster made for point ",clustered_p_1 ,"and ",clustered_p_2)

linkage_dictionary[11]=cluster_1

# Making each row for linkage matrix 
# each row is of the format as described here https://stackoverflow.com/questions/9838861/scipy-linkage-format
linkage_row=np.array([clustered_p_1,clustered_p_2,minimum_distance,len(linkage_dictionary[11])]).reshape(4,1).T

linkage_matrix=np.append(linkage_matrix,linkage_row,axis=0)

print(linkage_row,linkage_row.shape)



# actual_1=arr[clustered_p_1].reshape(1,2)

# actual_2=arr[clustered_p_2].reshape(1,2)

# arr=np.delete(arr,[clustered_p_1,clustered_p_2],axis=0) # removing rows that were clustered from original array


# print("#####")


# new_array=np.append(new_array,actual_2,axis=0) # forming new array 1. appending one of the two points that were clustered

# arr=np.concatenate((new_array,arr)) # appending those n-2 elements that were left from dataset

# new_proximity=proximity_matrix_gen(arr)


# # print(pd.DataFrame(new_proximity))

# print("#########")
# list_of_coordinate,minimum_distance = find_minimum_proximity(new_proximity)  # finding new minimum from new proximity matrix
# print(list_of_coordinate[0],minimum_distance)



# Have to think how can we utilise our previous calculations
# print(np.where(distance_arr[list_of_cordinates[0][0]] == infinite, 0, distance_arr[list_of_cordinates[0][0]]))
# print(np.where(distance_arr[list_of_cordinates[0][1]] == infinite, 0, distance_arr[list_of_cordinates[0][1]]))


# Important concepts

# In our proximity matrix there will be multiple combinations when we'll have mimimum distance but we'll take one at a time


# clustered_p_1 and clustered_p_2 corresponds to those 2 points who had contributed for least Euclidean distance 
# in proximity matrix