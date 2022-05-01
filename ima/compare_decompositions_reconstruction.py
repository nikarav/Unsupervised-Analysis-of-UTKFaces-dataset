from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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
from skimage import io, img_as_ubyte
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

# Does PCA improve classification accuracy?
# To test this, I will use a classifier, KNN.


k_inner = 3
k_outer = 5
cv_outer = KFold(n_splits=k_outer, shuffle=True, random_state=_SEED)

n_neighbors = [8, 10, 50, 85]

param_grid_knn = {
    'n_neighbors': n_neighbors
}


clf_error_knn = np.empty(k_outer)
clf_error_knn_pca_100 = np.empty(k_outer)
clf_error_knn_pca_500 = np.empty(k_outer)
clf_error_knn_pca_1000 = np.empty(k_outer)

n_neigh_opt = np.empty(k_outer)
n_neigh_opt_100 = np.empty(k_outer)
n_neigh_opt_500 = np.empty(k_outer)
n_neigh_opt_1000 = np.empty(k_outer)


for i, (train_idx, test_idx) in enumerate(cv_outer.split(X, y_gender)):
    print(f"Cross validation fold {i+1}/{k_outer}")
    y_train, y_test = y_gender[train_idx], y_gender[test_idx]
    X_train, X_test = X[train_idx, :], X[test_idx, :]

    scaler = preproc.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search_knn = GridSearchCV(KNeighborsClassifier(
    ), param_grid=param_grid_knn, cv=k_inner, verbose=2)

    grid_search_knn.fit(X_train, y_train)
    n_neigh_opt[i] = grid_search_knn.best_params_['n_neighbors']
    clf = grid_search_knn.best_estimator_.fit(X_train, y_train)
    clf_error_knn[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test))

    ############

    pca_100 = PCA(n_components=100)
    X_train_100 = pca_100.fit_transform(X_train)
    X_test_100 = pca_100.transform(X_test)

    grid_search_knn.fit(X_train_100, y_train)
    n_neigh_opt_100[i] = grid_search_knn.best_params_['n_neighbors']
    clf = grid_search_knn.best_estimator_.fit(X_train_100, y_train)
    clf_error_knn_pca_100[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test_100))

    ############

    pca_500 = PCA(n_components=500)
    X_train_500 = pca_500.fit_transform(X_train)
    X_test_500 = pca_500.transform(X_test)

    grid_search_knn.fit(X_train_500, y_train)
    n_neigh_opt_500[i] = grid_search_knn.best_params_['n_neighbors']
    clf = grid_search_knn.best_estimator_.fit(X_train_500, y_train)
    clf_error_knn_pca_500[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test_500))

    ############

    pca_1000 = PCA(n_components=1000)
    X_train_1000 = pca_1000.fit_transform(X_train)
    X_test_1000 = pca_1000.transform(X_test)

    grid_search_knn.fit(X_train_1000, y_train)
    n_neigh_opt_1000[i] = grid_search_knn.best_params_['n_neighbors']
    clf = grid_search_knn.best_estimator_.fit(X_train_1000, y_train)
    clf_error_knn_pca_1000[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test_1000))


mean_clf_err_knn = clf_error_knn.mean()
mean_clf_err_knn_pca_100 = clf_error_knn_pca_100.mean()
mean_clf_err_knn_pca_500 = clf_error_knn_pca_500.mean()
mean_clf_err_knn_pca_1000 = clf_error_knn_pca_1000.mean()

plt.title('Optimal number of neighbors')
plt.plot(np.arange(1, k_outer+1), n_neigh_opt, '1', label='KNN')
plt.plot(np.arange(1, k_outer+1), n_neigh_opt_100, '2', label='KNN/PCA 100')
plt.plot(np.arange(1, k_outer+1), n_neigh_opt_500, '3', label='KNN/PCA 500')
plt.plot(np.arange(1, k_outer+1), n_neigh_opt_1000, '4', label='KNN/PCA 1000')
plt.xlabel('Outer CV folds')
plt.ylabel('Number og neighbors')
plt.legend()
plt.show()


# What about 3 different classifiers? With n_components = 1000

param_grid_svc = {
    # 'C': [0.1, 1, 10],
    # 'kernel': ['rbf', 'linear']
}

param_grid_rf = {}

knn = KNeighborsClassifier()
svc = SVC()
rf = RandomForestClassifier()

pca = PCA(n_components=1000)


clf_error_knn_pca = np.empty(k_outer)
clf_error_svc_pca = np.empty(k_outer)
clf_error_rf_pca = np.empty(k_outer)

for i, (train_idx, test_idx) in enumerate(cv_outer.split(X, y_gender)):
    print(f"Outer cross validation loop {i+1}/{k_outer}")
    y_train, y_test = y_gender[train_idx], y_gender[test_idx]
    X_train, X_test = X[train_idx, :], X[test_idx, :]

    scaler = preproc.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ###############################################
    ######### PCA #############

    print('PCA')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print('KNN')
    clf = GridSearchCV(knn, param_grid=param_grid_knn, cv=k_inner, verbose=2)
    clf.fit(X_train_pca, y_train)
    clf = clf.best_estimator_
    clf.fit(X_train_pca, y_train)
    clf_error_knn_pca[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test_pca))

    print('SVC')
    clf = GridSearchCV(svc, param_grid=param_grid_svc, cv=k_inner, verbose=2)
    clf.fit(X_train_pca, y_train)
    clf = clf.best_estimator_
    clf.fit(X_train_pca, y_train)
    clf_error_svc_pca[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test_pca))

    print('RF')
    clf = GridSearchCV(rf, param_grid=param_grid_rf, cv=k_inner, verbose=2)
    clf.fit(X_train_pca, y_train)
    clf = clf.best_estimator_
    clf.fit(X_train_pca, y_train)
    clf_error_rf_pca[i] = ima_utils.classification_error(
        y_test, clf.predict(X_test_pca))

print('Done')

mean_knn_clf_err = clf_error_knn_pca.mean()
mean_svc_clf_err = clf_error_svc_pca.mean()
mean_rf_clf_err = clf_error_rf_pca.mean()


# SVC looks to provide the best results.  Find hyperparameters and compare different decompositions


# Compare reconstruction errors of the following methods, using different
# number of components. Methods:
# PCA (full), NMF, ICA, Dictionary Learning (Sparse).
# All other parameters random

n_components = [50, 100, 368, 500, 1000]

# Split data set to a train-val and test set. The train-val to be used in cross val. The test for a final comparison
X_train_val, X_test = train_test_split(X, test_size=0.15, random_state=_SEED)

k_outer = 3
cv_folds = KFold(n_splits=k_outer, shuffle=True, random_state=_SEED)


# Test no cv
scaler = preproc.StandardScaler(with_mean=False).fit(
    X_train_val)  # no mean for NMF
X_train_val_sc = scaler.transform(X_train_val)
X_test_sc = scaler.transform(X_test)


expl_var_pca = np.empty(len(n_components))

f_expl_var_pca_train = np.empty(len(n_components))
f_expl_var_nmf_train = np.empty(len(n_components))
f_expl_var_ica_train = np.empty(len(n_components))
f_expl_var_dl_train = np.empty(len(n_components))

f_rec_err_pca_train = np.empty(len(n_components))
f_rec_err_nmf_train = np.empty(len(n_components))
f_rec_err_ica_train = np.empty(len(n_components))
f_rec_err_dl_train = np.empty(len(n_components))

f_expl_var_pca_test = np.empty(len(n_components))
f_expl_var_nmf_test = np.empty(len(n_components))
f_expl_var_ica_test = np.empty(len(n_components))
f_expl_var_dl_test = np.empty(len(n_components))

f_rec_err_pca_test = np.empty(len(n_components))
f_rec_err_nmf_test = np.empty(len(n_components))
f_rec_err_ica_test = np.empty(len(n_components))
f_rec_err_dl_test = np.empty(len(n_components))

for i, n_comp in enumerate(n_components):
    print(f"Iteration {i+1}/{len(n_components)}, PCA")
    pca = PCA(n_components=n_comp).fit(X_train_val_sc)
    expl_var_pca[i] = np.sum(pca.explained_variance_ratio_)
    f_expl_var_pca_train[i] = ima_utils.get_decomposition_variance_score(
        pca, X_train_val_sc)
    f_expl_var_pca_test[i] = ima_utils.get_decomposition_variance_score(
        pca, X_test_sc)
    f_rec_err_pca_train[i] = ima_utils.get_decomposition_reconstruction_error_score(
        pca, X_train_val_sc)
    f_rec_err_pca_test[i] = ima_utils.get_decomposition_reconstruction_error_score(
        pca, X_test_sc)
    del pca

    print(f"Iteration {i+1}/{len(n_components)}, NMF")
    nmf = NMF(n_components=n_comp, max_iter=1000,
              solver='mu').fit(X_train_val_sc)
    f_expl_var_nmf_train[i] = ima_utils.get_decomposition_variance_score(
        nmf, X_train_val_sc)
    f_expl_var_nmf_test[i] = ima_utils.get_decomposition_variance_score(
        nmf, X_test_sc)
    f_rec_err_nmf_train[i] = ima_utils.get_decomposition_reconstruction_error_score(
        nmf, X_train_val_sc)
    f_rec_err_nmf_test[i] = ima_utils.get_decomposition_reconstruction_error_score(
        nmf, X_test_sc)
    del nmf

    print(f"Iteration {i+1}/{len(n_components)}, ICA")
    ica = FastICA(n_components=n_comp).fit(X_train_val_sc)
    f_expl_var_ica_train[i] = ima_utils.get_decomposition_variance_score(
        ica, X_train_val_sc)
    f_expl_var_ica_test[i] = ima_utils.get_decomposition_variance_score(
        ica, X_test_sc)
    f_rec_err_ica_train[i] = ima_utils.get_decomposition_reconstruction_error_score(
        ica, X_train_val_sc)
    f_rec_err_ica_test[i] = ima_utils.get_decomposition_reconstruction_error_score(
        ica, X_test_sc)
    del ica

    # print(f"Iteration {i+1}/{len(n_components)}, DL")
    # dl = DictionaryLearning(n_components=n_comp).fit(X_train_val_sc)
    # f_expl_var_dl_train[i] = ima_utils.get_decomposition_variance_score(
    #     dl, X_train_val_sc)
    # f_expl_var_dl_test[i] = ima_utils.get_decomposition_variance_score(
    #     dl, X_test_sc)
    # f_rec_err_dl_train[i] = ima_utils.get_decomposition_reconstruction_error_score(
    #     dl, X_train_val_sc)
    # f_rec_err_dl_test[i] = ima_utils.get_decomposition_reconstruction_error_score(
    #     dl, X_test_sc)
    # del dl

threshold = 0.95

idx_pca_train = np.where(f_expl_var_pca_train >= threshold)[0]
idx_pca_test = np.where(f_expl_var_pca_test >= threshold)[0]

idx_nmf_train = np.where(f_expl_var_nmf_train >= threshold)[0]
idx_nmf_test = np.where(f_expl_var_nmf_test >= threshold)[0]

idx_ica_train = np.where(f_expl_var_ica_train >= threshold)[0]
idx_ica_test = np.where(f_expl_var_ica_test >= threshold)[0]

threshold_loss = 0.1
idx_pca_rec_train = np.where(f_rec_err_pca_train <= threshold_loss)[0]
idx_pca_rec_test = np.where(f_rec_err_pca_train <= threshold_loss)[0]

idx_nmf_rec_train = np.where(f_rec_err_nmf_train <= threshold_loss)[0]
idx_nmf_rec_test = np.where(f_rec_err_nmf_test <= threshold_loss)[0]

idx_ica_rec_train = np.where(f_rec_err_ica_train <= threshold_loss)[0]
idx_ica_rec_test = np.where(f_rec_err_ica_test <= threshold_loss)[0]

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(n_components, f_expl_var_pca_train, label='PCA train')
plt.plot(n_components, f_expl_var_pca_test, label='PCA test')
plt.plot([n_components[0], n_components[-1]], [f_expl_var_pca_train[idx_pca_train],
         f_expl_var_pca_train[idx_pca_train]], 'k--', label=f'{threshold*100}% threshold train')
plt.plot([n_components[0], n_components[-1]], [f_expl_var_pca_test[idx_pca_test],
         f_expl_var_pca_test[idx_pca_test]], 'g--', label=f'{threshold*100}% threshold test')
# plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_components, f_rec_err_pca_train, label='PCA train')
plt.plot(n_components, f_rec_err_pca_test, label='PCA test')
plt.plot([n_components[0], n_components[-1]], [f_rec_err_pca_train[idx_pca_rec_train],
         f_rec_err_pca_train[idx_pca_rec_train]], 'k--', label=f'{threshold*100}% threshold train')
plt.plot([n_components[0], n_components[-1]], [f_rec_err_pca_test[idx_pca_rec_test],
         f_rec_err_pca_test[idx_pca_rec_test]], 'g--', label=f'{threshold*100}% threshold test')
plt.legend()
plt.plot()

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(n_components, f_expl_var_nmf_train, label='NMF train')
plt.plot(n_components, f_expl_var_nmf_test, label='NMF test')
plt.plot([n_components[0], n_components[-1]], [f_expl_var_nmf_train[idx_nmf_train],
         f_expl_var_nmf_train[idx_nmf_train]], 'k--', label=f'{threshold*100}% threshold train')
plt.plot([n_components[0], n_components[-1]], [f_expl_var_nmf_test[idx_nmf_test],
         f_expl_var_nmf_test[idx_nmf_test]], 'g--', label=f'{threshold*100}% threshold test')
# plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_components, f_rec_err_nmf_train, label='NMF train')
plt.plot(n_components, f_rec_err_nmf_test, label='NMF test')
plt.plot([n_components[0], n_components[-1]], [f_rec_err_nmf_train[idx_nmf_rec_train],
         f_rec_err_nmf_train[idx_nmf_rec_train]], 'k--', label=f'{threshold*100}% threshold train')
plt.plot([n_components[0], n_components[-1]], [f_rec_err_nmf_test[idx_nmf_rec_test],
         f_rec_err_nmf_test[idx_nmf_rec_test]], 'g--', label=f'{threshold*100}% threshold test')
plt.legend()
plt.plot()

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(n_components, f_expl_var_ica_train, label='ICA train')
plt.plot(n_components, f_expl_var_ica_test, label='ICA test')
plt.plot([n_components[0], n_components[-1]], [f_expl_var_ica_train[idx_ica_train],
         f_expl_var_ica_train[idx_pca_train]], 'k--', label=f'{threshold*100}% threshold train')
plt.plot([n_components[0], n_components[-1]], [f_expl_var_ica_test[idx_ica_test],
         f_expl_var_ica_test[idx_pca_test]], 'g--', label=f'{threshold*100}% threshold test')
# plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_components, f_rec_err_ica_train, label='ICA train')
plt.plot(n_components, f_rec_err_ica_test, label='ICA test')
plt.plot([n_components[0], n_components[-1]], [f_rec_err_ica_train[idx_ica_rec_train],
         f_rec_err_ica_train[idx_ica_rec_train]], 'k--', label=f'{threshold*100}% threshold train')
plt.plot([n_components[0], n_components[-1]], [f_rec_err_ica_test[idx_ica_rec_test],
         f_rec_err_ica_test[idx_ica_rec_test]], 'g--', label=f'{threshold*100}% threshold test')
plt.legend()
plt.plot()


plt.figure(figsize=(15, 8))
plt.title('Explained variance train set')
plt.plot(n_components, f_expl_var_pca_train, label='PCA')
plt.plot(n_components, f_expl_var_nmf_train, label='NMF')
plt.plot(n_components, f_expl_var_ica_train, label='ICA')
plt.legend()
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()


plt.figure(figsize=(15, 8))
plt.title('Explained variance test set')
plt.plot(n_components, f_expl_var_pca_test, label='PCA')
plt.plot(n_components, f_expl_var_nmf_test, label='NMF')
plt.plot(n_components, f_expl_var_ica_test, label='ICA')
plt.legend()
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()


# Let's say we want to have a fixed hyperparameter classifier, eg KNN
# KNN is a good classifier, since it does not require many parameters to tune
# It is however affected by high dimensionalty
