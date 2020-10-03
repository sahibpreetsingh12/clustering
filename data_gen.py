from sklearn.datasets import make_blobs,make_circles,make_moons,make_swiss_roll
import numpy as np



n_samples=[100,200,300]

feature_list=[2,3,4]

datasets= []

dataset_creator=[make_circles,make_moons,make_swiss_roll]

for iter,features in zip(n_samples,feature_list):
    X,y = make_blobs(n_samples=iter,n_features=features,centers=3)
    datasets.append(X)  # appending datasets genrated from make_blobs

for creator in dataset_creator:
    for iter in n_samples:
        X,y=creator(n_samples=iter)
        datasets.append(X)

print(len(datasets))