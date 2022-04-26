from pickletools import uint8
import ima_utils
import sklearn.preprocessing as preproc
from sklearn.decomposition import PCA, FastICA, NMF, DictionaryLearning
import scipy.linalg as lng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from tqdm import tqdm
import os


# Globals
_RANDOM_STATE = 0
_LOAD_DATA = False
_LOAD_ALL = False
_SAVE = False

# Set the paths for image files, labels etc
faces_path = "../data/Faces/"
labels_path = "../data/labels.csv"
names_path = "../data/filenames.txt"
data_set_npy = 'data.npy'
labels_csv_path = 'data_labels.csv'
# Get labels as pandas object from labels.csv
labels = ima_utils.get_labels_df(
    labels_path=labels_path, names_path=names_path)

# Put age in bins
age_bin_width_years = 10
ages = labels.age.values//age_bin_width_years  # bin_width fully lived
labels.drop(['age'], axis=1, inplace=True)
labels['age_bin'] = ages


# Open a random image to get dimensions
h, w = ima_utils.get_dimensions_from_an_image(faces_path, 0, as_gray=True)


# Get images in data matrix
# Row major, ie create a row vector by concatenating each row to the previous one, from left to right

if _LOAD_DATA:
    _LOAD_ALL = False

if _LOAD_ALL:
    dirs = os.listdir(faces_path)
    X = np.empty((len(dirs), h*w), dtype=np.ubyte)

    for i in tqdm(range(len(dirs))):
        a = io.imread(faces_path+f"{i}.jpg", as_gray=True)
        a = img_as_ubyte(a)
        X[i, :] = a.reshape(1, -1)
    del a
    labels_loaded = labels.copy()
    if _SAVE:
        np.save(data_set_npy, X)
        labels_loaded = labels.copy()
        labels_loaded.to_csv(labels_csv_path, index=False)

if _LOAD_DATA and (os.path.isfile(data_set_npy) and os.path.isfile(labels_csv_path)):
    X = np.load(data_set_npy)
    labels_loaded = pd.read_csv(labels_csv_path)
else:
    n_elements_from_label = 5000
    label_to_choose_from = 'gender'
    images_to_load = ima_utils.pick_n_from_label(
        labels, n_elements_from_label, label_to_choose_from, shuffle=False)

    X = np.empty((len(images_to_load), h*w), dtype=np.ubyte)

    for i in tqdm(range(len(images_to_load))):
        a = io.imread(faces_path+f"{images_to_load[i]}.jpg", as_gray=True)
        a = img_as_ubyte(a)
        X[i, :] = a.reshape(1, -1)

    del a
    labels_loaded = labels.loc[images_to_load]
