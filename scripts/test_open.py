import os
import pathlib
import pandas as pd
import numpy as np
from skimage import io
import sklearn.preprocessing as preproc
import matplotlib.pyplot as plt

data_path = 'data/Faces/'
labels_path = 'data/labels.csv'

labels = pd.read_csv(labels_path)
labels_columns = ['age', 'gender', 'race']
labels = pd.DataFrame(data=labels.values, columns=labels_columns)

dirs = os.listdir(data_path)
no_of_images = len(dirs)

# Open one image to get dimensions
k = 350
img_rnd = io.imread(data_path+f'{k}.jpg')
h,w,_ = img_rnd.shape
del img_rnd

batch_ratio = 10
bathc_len = no_of_images//batch_ratio
current_index=0



## Create the binary. Open it in append mode

k = 100
X = np.empty((k,h*w))


for i in range(k):
    print(f'Loading image {i+1}/{k}')
    a = io.imread(data_path+f'{i}.jpg',as_gray=True)
    X[i,:] = a.reshape(1,-1)
    del a


import scipy.linalg as lng

U,S,Vh = lng.svd(X, full_matrices=False)
V = Vh.T
del Vh

Z = U*S

cumsum = np.cumsum(S*S/(S*S).sum())

threshold = 0.9

def plot_eigenface(V,eigenface=1):
    i = eigenface-1
    plt.subplot(1,2,1)
    plt.imshow( (V[:,i]).reshape(h,w), cmap='gray')
    plt.xticks(())
    plt.yticks(())

    plt.subplot(1,2,2)
    plt.imshow((-V[:,i]).reshape(h,w), cmap='gray')
    plt.xticks(())
    plt.yticks(())

    plt.show()
    