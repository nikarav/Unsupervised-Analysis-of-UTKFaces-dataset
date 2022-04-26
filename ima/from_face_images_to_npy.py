import ima_utils
import os
import numpy as np
from skimage import io, img_as_ubyte
from tqdm import tqdm


# Set the paths for image files, labels etc
faces_path = "../data/Faces/"
labels_path = "../data/labels.csv"
names_path = "../data/filenames.txt"
data_set_gray_npy = 'data_gray.npy'
data_set_rgb_npy = 'data_rgb.npy'
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

# Save grayscale images
dirs = os.listdir(faces_path)
X = np.empty((len(dirs), h*w), dtype=np.ubyte)

for i in tqdm(range(len(dirs))):
    a = io.imread(faces_path+f"{i}.jpg", as_gray=True)
    a = img_as_ubyte(a)
    X[i, :] = a.reshape(1, -1)
del a
labels_loaded = labels.copy()

# Save labels and data matrix X to disk
np.save(data_set_gray_npy, X)
labels_loaded = labels.copy()
labels_loaded.to_csv(labels_csv_path, index=False)

del X


X_rgb = np.empty((len(dirs), h*w, 3), dtype=np.ubyte)

for i in tqdm(range(len(dirs))):
    a = io.imread(faces_path+f"{i}.jpg")
    X_rgb[i, :, :] = a.reshape(1, -1, 3)
del a
# Save data to disk
np.save(data_set_rgb_npy, X_rgb)
del X_rgb
