from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from pickletools import uint8
import ima_utils
import sklearn.preprocessing as preproc
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA, FastICA, NMF, DictionaryLearning, IncrementalPCA
import scipy.linalg as lng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Globals
_SHUFFLE = True
_SEED = 0

# Set the paths for image files, labels etc
faces_path = "../data/Faces/"
labels_path = "../data/labels.csv"
names_path = "../data/filenames.txt"

data_set_gray_npy = "data_gray.npy"
data_set_rgb_npy = "data_rgb.npy"
labels_csv_path = "data_labels.csv"
# Get labels as pandas object from labels.csv
labels = ima_utils.get_labels_df(
    labels_path=labels_path, names_path=names_path)

# Put age in bins
age_bin_width_years = 25
ages = labels.age.values//age_bin_width_years  # bin_width fully lived
labels.drop(['age'], axis=1, inplace=True)
labels['age_bin'] = ages


# Open a random image to get dimensions
h, w = ima_utils.get_dimensions_from_an_image(faces_path, 0, as_gray=True)


n_elements_from_label = 1000
label_to_choose_from = "gender"
images_to_load = ima_utils.pick_n_from_label(
    labels, n_elements_from_label, label_to_choose_from, shuffle=_SHUFFLE
)

X = ima_utils.read_images_from_npy(data_set_gray_npy, images_to_load)


labels_loaded = labels.loc[images_to_load]
labels_loaded = labels_loaded.reset_index()

y_age = labels_loaded.age_bin.values
y_race = labels_loaded.race.values
y_gender = labels_loaded.gender.values

############################################################
