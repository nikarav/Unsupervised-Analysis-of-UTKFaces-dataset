from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import enum
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
    age_max = 45
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

##################################

max_iter = 1000

n_components_pca = [100, 300]
n_components_ica = [50, 75, 100]
n_components_nmf = [50, 75, 100]

n_neighbors = [5, 10, 20, 50]

grid_param_knn = {
    'n_neighbors': n_neighbors
}

knn_pca = Pipeline([
    ('decomp', PCA()),
    ('clf', KNeighborsClassifier())
])

knn_ica = Pipeline([
    ('decomp', FastICA(max_iter=max_iter)),
    ('clf', KNeighborsClassifier())
])


knn_nmf = Pipeline([
    ('decomp', NMF(init='random', max_iter=max_iter)),
    ('clf', KNeighborsClassifier())
])


grid_param_knn_pca = {
    'decomp__n_components': n_components_pca,
    'clf__n_neighbors': n_neighbors
}


grid_param_knn_ica = {
    'decomp__n_components': n_components_ica,
    'clf__n_neighbors': n_neighbors
}


grid_param_knn_nmf = {
    'decomp__n_components': n_components_nmf,
    'clf__n_neighbors': n_neighbors
}


k_outer = 2
k_inner = 2

cv_outer = KFold(n_splits=k_outer, shuffle=_SHUFFLE, random_state=_SEED)

gender_accuracy_knn = np.empty(k_outer)
gender_accuracy_knn_pca = np.empty(k_outer)
gender_accuracy_knn_ica = np.empty(k_outer)
gender_accuracy_knn_nmf = np.empty(k_outer)

age_accuracy_knn = np.empty(k_outer)
age_accuracy_knn_pca = np.empty(k_outer)
age_accuracy_knn_ica = np.empty(k_outer)
age_accuracy_knn_nmf = np.empty(k_outer)

race_accuracy_knn = np.empty(k_outer)
race_accuracy_knn_pca = np.empty(k_outer)
race_accuracy_knn_ica = np.empty(k_outer)
race_accuracy_knn_nmf = np.empty(k_outer)

for i, (train_idx, test_idx) in enumerate(cv_outer.split(X, y_age)):
    print(f'Outer CV loop {i+1}/{k_outer}')
    y_age_train, y_age_test = y_age[train_idx], y_age[test_idx]
    y_gender_train, y_gender_test = y_gender[train_idx], y_gender[test_idx]
    y_race_train, y_race_test = y_race[train_idx], y_race[test_idx]

    X_train, X_test = X[train_idx, :], X[test_idx, :]

    grid_knn = GridSearchCV(KNeighborsClassifier(),
                            param_grid=grid_param_knn, cv=k_inner)
    grid_knn_pca = GridSearchCV(
        knn_pca, param_grid=grid_param_knn_pca, cv=k_inner)
    grid_knn_ica = GridSearchCV(
        knn_ica, param_grid=grid_param_knn_ica, cv=k_inner)
    grid_knn_nmf = GridSearchCV(
        knn_nmf, param_grid=grid_param_knn_nmf, cv=k_inner)

    # Gender
    txt = 'Gender'
    print(f'Finding best parameters for {txt} KNN')
    grid_knn.fit(X_train, y_gender_train)
    print(f'Finding best parameters for {txt} PCA KNN')
    grid_knn_pca.fit(X_train, y_gender_train)
    print(f'Finding best parameters for {txt} ICA KNN')
    grid_knn_ica.fit(X_train, y_gender_train)
    print(f'Finding best parameters for {txt} NMF KNN')
    grid_knn_nmf.fit(X_train, y_gender_train)

    grid_knn.best_estimator_.fit(X_train, y_gender_train)
    grid_knn_pca.best_estimator_.fit(X_train, y_gender_train)
    grid_knn_ica.best_estimator_.fit(X_train, y_gender_train)
    grid_knn_nmf.best_estimator_.fit(X_train, y_gender_train)

    gender_accuracy_knn[i] = grid_knn.best_estimator_.score(
        X_test, y_gender_test)
    gender_accuracy_knn_pca[i] = grid_knn_pca.best_estimator_.score(
        X_test, y_gender_test)
    gender_accuracy_knn_ica[i] = grid_knn_ica.best_estimator_.score(
        X_test, y_gender_test)
    gender_accuracy_knn_nmf[i] = grid_knn_nmf.best_estimator_.score(
        X_test, y_gender_test)


print('Done')
