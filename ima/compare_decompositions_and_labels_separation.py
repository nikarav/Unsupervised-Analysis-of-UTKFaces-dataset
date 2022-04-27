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
age_bin_width_years = 10
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

X = np.empty((len(images_to_load), h * w), dtype=np.ubyte)

for i in tqdm(range(len(images_to_load))):
    a = io.imread(faces_path + f"{images_to_load[i]}.jpg", as_gray=True)
    a = img_as_ubyte(a)
    X[i, :] = a.reshape(1, -1)
del a
labels_loaded = labels.loc[images_to_load]
labels_loaded = labels_loaded.reset_index()


# # pca = IncrementalPCA(n_components=400, batch_size=400)
# pca = PCA(n_components=400)
# pca.fit(X)

# X_transf = pca.transform(X)


# X_transf = pd.DataFrame(data=X_transf, index=labels_loaded.index)
# y_df = labels_loaded[[label_to_choose_from]]

# ima_utils.plot_2d_with_labels(X_transf, y_df, decomposition_method='PCA')


ica = FastICA(n_components=25)
ica.fit(X)

X_transf = ica.transform(X)
X_transf = pd.DataFrame(data=X_transf, index=labels_loaded.index)
y_df = labels_loaded[[label_to_choose_from]]

ima_utils.plot_2d_with_labels(X_transf, y_df, components=[
                              0, 3], decomposition_method='PCA')


v = ica.components_.T
