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
age_bin_width_years = 10
ages = labels.age.values//age_bin_width_years  # bin_width fully lived
labels.drop(['age'], axis=1, inplace=True)
labels['age_bin'] = ages


# Open a random image to get dimensions
h, w = ima_utils.get_dimensions_from_an_image(faces_path, 0, as_gray=True)


n_elements_from_label = 2000
label_to_choose_from = "random"
images_to_load = ima_utils.pick_n_from_label(
    labels, n_elements_from_label, label_to_choose_from, shuffle=_SHUFFLE
)

X = np.empty((len(images_to_load), h * w), dtype=np.ubyte)

for i in tqdm(range(len(images_to_load))):
    a = io.imread(faces_path + f"{images_to_load[i]}.jpg", as_gray=True)
    a = img_as_ubyte(a)
    X[i, :] = a.reshape(1, -1)
del a
labels_loaded = labels.loc[images_to_load]
labels_loaded = labels_loaded.reset_index()


k_folds = 5
cv = KFold(n_splits=k_folds)

y_age = labels_loaded.age_bin.values
y_race = labels_loaded.race.values
y_gender = labels_loaded.gender.values

############################################################


n_components = [350, 400, 450, 500, 550]
n_neighbors = [5, 10, 15, 20]


pca_clf = Pipeline([
    ('decomp', PCA()),
    ('clf', RandomForestClassifier())
])

param_grid = {
    'decomp__n_components': n_components,
    # 'clf__n_neighbors': n_neighbors
}

grid_search = GridSearchCV(
    pca_clf, param_grid=param_grid, cv=k_folds, verbose=2)
grid_search.fit(X, y_age)

#############################################################

acc_age = np.empty((k_folds, len(n_components)))
acc_race = np.empty((k_folds, len(n_components)))
acc_gender = np.empty((k_folds, len(n_components)))

acc_age_no = np.empty(k_folds)
acc_race_no = np.empty(k_folds)
acc_gender_no = np.empty(k_folds)

clf = KNeighborsClassifier(n_neighbors=n_neighbors)
# clf = SVC()
for i, (train_idx, test_idx) in enumerate(cv.split(X, y_age)):
    y_age_train, y_age_test = y_age[train_idx], y_age[test_idx]
    y_race_train, y_race_test = y_race[train_idx], y_race[test_idx]
    y_gender_train, y_gender_test = y_gender[train_idx], y_gender[test_idx]

    X_train, X_test = X[train_idx, :], X[test_idx, :]

    clf.fit(X_train, y_age_train)
    acc_age_no[i] = clf.score(X_test, y_age_test)
    clf.fit(X_train, y_race_train)
    acc_race_no[i] = clf.score(X_test, y_race_test)
    clf.fit(X_train, y_gender_train)
    acc_gender_no[i] = clf.score(X_test, y_gender_test)

    for j, n_comp in tqdm(enumerate(n_components)):
        pca = PCA(n_components=n_comp, copy=True)
        X_train_transf = pca.fit_transform(X_train)
        X_test_transf = pca.transform(X_test)
        clf.fit(X_train_transf, y_age_train)
        acc_age[i, j] = clf.score(X_test_transf, y_age_test)
        clf.fit(X_train_transf, y_race_train)
        acc_race[i, j] = clf.score(X_test_transf, y_race_test)
        clf.fit(X_train_transf, y_gender_train)
        acc_gender[i, j] = clf.score(X_test_transf, y_gender_test)


idx_age = np.argmax(acc_age.mean(axis=0))
idx_race = np.argmax(acc_race.mean(axis=0))
idx_gender = np.argmax(acc_gender.mean(axis=0))

n_components_age = n_components[idx_age]
n_components_race = n_components[idx_race]
n_components_gender = n_components[idx_gender]
