from sklearn.model_selection import GridSearchCV
from pickletools import uint8
import ima_utils
import sklearn.preprocessing as preproc
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA, FastICA, NMF, DictionaryLearning, IncrementalPCA
import scipy.linalg as lng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Globals
_SHUFFLE = True
_SEED = 0

_LIMITS = True

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


# Limit age
if _LIMITS:
    age_min = 18
    age_max = 80
    labels = labels.loc[(labels.age >= age_min) & (labels.age <= age_max)]


# Put age in bins
age_bin_width_years = 25
ages = labels.age.values//age_bin_width_years  # bin_width fully lived
labels.drop(['age'], axis=1, inplace=True)
labels['age_bin'] = ages


train_dataset_idx, test_dataset_idx = train_test_split(labels.index.to_list())


# Open a random image to get dimensions
h, w = ima_utils.get_dimensions_from_an_image(faces_path, 0, as_gray=True)

X = np.load(data_set_gray_npy)

data_train, data_test = X[train_dataset_idx], X[test_dataset_idx]
labels_train, labels_test = labels.loc[train_dataset_idx], labels.loc[test_dataset_idx]
labels_train = labels_train.reset_index()
labels_test = labels_test.reset_index()
del X, labels  # to avoid accessing the test set and also to save ram


n_elements_from_label = 1500
label_to_choose_from = "random"
images_to_load = ima_utils.pick_n_from_label(
    labels_train, n_elements_from_label, label_to_choose_from, shuffle=_SHUFFLE
)

X = data_train[images_to_load, :]


labels_loaded = labels_train.loc[images_to_load]
labels_loaded = labels_loaded.reset_index()

y_age = labels_loaded.age_bin.values
y_race = labels_loaded.race.values
y_gender = labels_loaded.gender.values

###################################################
# Do decompositions to the dataset we have loaded #
###################################################

X_train, X_test = train_test_split(X)

n_components = 20
max_iter = 500

scaler = preproc.StandardScaler(with_mean=False)
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

nmf = NMF(n_components=n_components, solver='mu', max_iter=max_iter,
          random_state=_SEED, init='random').fit(X_train_sc)

X_train_nmf = nmf.transform(X_train_sc)
X_test_nmf = nmf.transform(X_test_sc)


fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, (component, ax) in enumerate(zip(nmf.components_[:20], axes.ravel())):
    ax.imshow(component.reshape(h, w), cmap='gray')
    ax.set_title(f"{i+1}. component")


# Look at specific components

specific_component = 4
if specific_component > n_components:
    specific_component = n_components

idx = np.argsort(X_train_nmf[:, (specific_component-1)])
idx = (idx[::-1])
idx = idx[:5]

ima_utils.plot_image(nmf.components_[(specific_component-1)], h, w)
plt.show()
ima_utils.plot_image(X_train_sc[idx[0], :], h, w)
plt.show()
ima_utils.plot_image(X_train_sc[idx[1], :], h, w)
plt.show()
ima_utils.plot_image(X_train_sc[idx[2], :], h, w)
plt.show()


# Find a good number of components?

k_folds = 5
cv = KFold(n_splits=k_folds)

rec_loss = np.empty(k_folds)

for i, (train_idx, test_idx) in enumerate(cv.split((X_train_sc))):
    X_tr, X_tst = X_train_sc[train_idx], X_train_sc[test_idx]
