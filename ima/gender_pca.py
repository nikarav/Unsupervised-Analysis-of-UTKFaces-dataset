from pickletools import uint8
import ima_utils
import sklearn.preprocessing as preproc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA, NMF, DictionaryLearning, IncrementalPCA
import scipy.linalg as lng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from tqdm import tqdm
import os


# Globals
_SHUFFLE = True

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
# age_bin_width_years = 10
# ages = labels.age.values//age_bin_width_years  # bin_width fully lived
# labels.drop(['age'], axis=1, inplace=True)
# labels['age_bin'] = ages


# Open a random image to get dimensions
h, w = ima_utils.get_dimensions_from_an_image(faces_path, 0, as_gray=True)


# Limit images to choose from using some criterion
age_min = 18
age_max = 75
labels_limited = labels.loc[(labels.age > age_min) & (labels.age < age_max)]

n_elements_from_label = 1500
label_to_choose_from = "gender"
images_to_load = ima_utils.pick_n_from_label(
    labels_limited, n_elements_from_label, label_to_choose_from, shuffle=_SHUFFLE
)

X = np.empty((len(images_to_load), h * w), dtype=np.ubyte)

for i in tqdm(range(len(images_to_load))):
    a = io.imread(faces_path + f"{images_to_load[i]}.jpg", as_gray=True)
    a = img_as_ubyte(a)
    X[i, :] = a.reshape(1, -1)
del a
labels_loaded = labels.loc[images_to_load]
labels_loaded = labels_loaded.reset_index()


# Split to male and female gender data sets
labels_male = labels_loaded.loc[labels_loaded.gender == 0]
labels_female = labels_loaded.loc[labels_loaded.gender == 1]

X_male, X_female = X[labels_male.index.tolist(
), :], X[labels_female.index.to_list(), :]

labels_male = labels_male.reset_index()
labels_female = labels_female.reset_index()


# Apply PCA to each of the datasets

X_male_train, X_male_test = train_test_split(
    X_male, test_size=0.1, shuffle=_SHUFFLE)
X_female_train, X_female_test = train_test_split(
    X_female, test_size=0.1, shuffle=_SHUFFLE)

del X_male, X_female

pca_male = PCA(n_components=0.99)
X_male_transf = pca_male.fit_transform(X_male_train)
v_male = pca_male.components_.T

pca_female = PCA(n_components=0.99)
X_female_transf = pca_female.fit_transform(X_female_train)
v_female = pca_female.components_.T


test_k = 15
test_male = X_male_test[test_k]
transformed_to_female = ((test_male - pca_male.mean_) @ v_female).dot(
    v_female.T
) + pca_female.mean_
reconstructed_male = ((test_male - pca_male.mean_) @ v_male).dot(
    v_male.T
) + pca_male.mean_

plt.figure()
plt.subplot(1, 2, 1)
ima_utils.plot_image(reconstructed_male, h, w)
plt.subplot(1, 2, 2)
ima_utils.plot_image(transformed_to_female, h, w)
