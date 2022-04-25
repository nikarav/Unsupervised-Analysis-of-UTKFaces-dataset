from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from skimage import io, img_as_ubyte
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preproc

import seaborn as sns

# Globals
_SEED = 0

# Set the paths for image files, labels etc
faces_path = "../data/Faces/"
labels_path = "../data/labels.csv"
names_path = "../data/filenames.txt"

# Read labels as pandas object from labels.csv
labels_columns = ['age', 'gender', 'race']
labels = pd.read_csv(labels_path)
labels = pd.DataFrame(data=labels.values, columns=labels_columns)

# Print initial statistics regarding the data set and the labels we have

# Inspect for nan's
# print(labels.info())

# Print count of unique values for each label
print(f"Gender:\n{labels.gender.value_counts()}")
print()
print(f"Race:\n{labels.race.value_counts()}")

# If we want to ignore the "other" or 4, race values, run the following
# We can ignore them in the grounds of the sample being smaller than the others.
# labels = labels[labels.race!=4]

# Put age in bins
age_bin_width_years = 10
ages = labels.age.values//age_bin_width_years  # bin_width fully lived
labels.drop(['age'], axis=1, inplace=True)
labels['age_bin'] = ages

################################################

# We might want to remove some observations from race 0, since it has about twice
# the number of observations of each other race, apart from 'other'

##################################################
# Randomly pick 500 obs of each race. I should also prob have qual obs with age and genders
# This is an initial exploration


def pick_n_from_label(n, label):
    '''
    Returns list of image names to load.
    n: int, Number of elements from each label
    label: string, Column from the pandas object to pick from its elements
    '''
    lbl = label.lower()
    image_names_to_load = []
    for i in labels[lbl].unique():
        indices = labels[labels[lbl] == i].index.to_list()
        np.random.seed(_SEED)
        np.random.shuffle(indices)
        try:
            image_names_to_load += indices[:n]
        except:
            image_names_to_load += indices
    return image_names_to_load


# Open a random image to get dimensions
test_img_name = 0
test_img = io.imread(faces_path+f"{test_img_name}.jpg", as_gray=True)
h, w = test_img.shape
del test_img

n_elements_from_label = 300
label_to_choose_from = 'race'

images_to_load = pick_n_from_label(n_elements_from_label, label_to_choose_from)


X = np.empty((len(images_to_load), h*w))

h_c, w_c = 100, 100  # cropped image height and width
X_c = np.empty((len(images_to_load), h_c*w_c))

for i in range(len(images_to_load)):
    print(f"{i+1}/{len(images_to_load)}")
    a = io.imread(faces_path+f"{images_to_load[i]}.jpg", as_gray=True)
    a = img_as_ubyte(a)
    X[i, :] = a.reshape(1, -1)
    a = resize(a, (h_c, w_c))
    X_c[i, :] = a.reshape(1, -1)

del a
print('Done loading images')

X = X.astype(int)
# np.save(f"X_{label_to_choose_from}_{n_elements_from_label}.npy", X)
labels_loaded = labels.loc[images_to_load]

y = labels_loaded['race'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=_SEED)
X_c_train, X_c_test, _, _ = train_test_split(X, y, random_state=_SEED)

pca = PCA(n_components=0.95, random_state=_SEED)

X_train_pca = pca.fit_transform(X_train/255.)
X_test_pca = pca.transform(X_test/255.)

X_c_train_pca = pca.fit_transform(X_c_train/255.)
X_c_test_pca = pca.transform(X_c_test/255.)


scaler = preproc.StandardScaler()
X_train = scaler.fit_transform(X_train/255.)
X_test = scaler.transform(X_test/255.)

X_c_train = scaler.fit_transform(X_c_train/255.)
X_test = scaler.transform(X_c_test/255.)

classif = LogisticRegression()

classif.fit(X_train, y_train)
scores_200 = classif.score(X_test, y_test)

classif.fit(X_c_train, y_train)
scores_100 = classif.score(X_c_test, y_test)

classif.fit(X_train_pca, y_train)
scores_200_pca = classif.score(X_test_pca, y_test)

classif.fit(X_c_train_pca, y_train)
scores_100_pca = classif.score(X_c_test_pca, y_test)

plt.subplot(1, 2, 1)
plt.imshow(X[0, :].reshape(h, w), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(X_c[0, :].reshape(h_c, w_c), cmap='gray')
