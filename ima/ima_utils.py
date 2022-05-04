import numpy as np
import pandas as pd
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.linalg as lng


def plot_image(vector, height, width, cmap='gray'):
    dims = len(vector.shape)
    if dims == 3:
        plt.imshow(vector.reshape(height, width, 3))
        return
    plt.imshow(vector.reshape(height, width), cmap=cmap)


def plot_2d_with_labels(X_df, label_df, components=[0, 1], decomposition_method=''):
    import seaborn as sns
    if len(components) > 2:
        print('It is a 2d plot. Only 2 components required')
        return
    df = pd.DataFrame(
        data=X_df.loc[:, [components[0], components[1]]], index=X_df.index)
    df = pd.concat((df, label_df), axis=1, join='inner')
    df.columns = [f'Comp. {components[0]+1} vector',
                  f'Comp. {components[1]+1} vector',
                  f'{label_df.columns.to_list()[0]}']
    sns.lmplot(x=df.columns.to_list()[0], y=df.columns.to_list()[1],
               hue=df.columns.to_list()[2], data=df, fit_reg=False)
    ax = plt.gca()
    ax.set_title('Observations using ' + decomposition_method)


def get_dimensions_from_an_image(faces_path, image_no=0, as_gray=True):
    '''
    Returns 2 integers, height (h) and width (w) as dimensions of an image
    '''
    test_img = io.imread(faces_path+f"{image_no}.jpg", as_gray=as_gray)
    if as_gray:
        h, w = test_img.shape
        return h, w
    h, w, _ = test_img.shape
    return h, w


def pick_n_from_label(df, n, label, random_state=0, shuffle=True):
    '''
    Returns list of image names to load.
    n: int, Number of elements from each label
    label: string, Column from the pandas object to pick from its elements
    If label == 'random', then n random images are loaded
    '''
    labels = df.copy()
    lbl = label.lower()
    image_names_to_load = []
    if lbl == 'random':
        indices = labels.index.to_list()
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        try:
            image_names_to_load += indices[:n]
        except IndexError:
            image_names_to_load += indices
        return image_names_to_load
    for i in labels[lbl].unique():
        indices = labels[labels[lbl] == i].index.to_list()
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        try:
            image_names_to_load += indices[:n]
        except IndexError:
            image_names_to_load += indices
    return image_names_to_load


def get_labels_df(labels_path, names_path=None):
    '''
    Return a pandas data frame with the following columns:
        Index (automatically generated). Can be useful if we need to reset it later.
        Processed_image filename as in the Faces folder, without the .jpg filetype extension
        Image actual file name, if names_path is not None or empty.
        Age: int, positive
        Gender: binary ,0: male, 1: female
        Race: int. 0:white, 1:black, 2:asian, 3:indian, 4:other
    '''
    labels_columns = ['age', 'gender', 'race']
    labels = pd.read_csv(labels_path, header=None)
    labels = pd.DataFrame(data=labels.values, columns=labels_columns)
    labels['image_no'] = labels.index
    if names_path is not None and names_path != '':
        names = []
        with open(names_path, 'r') as f:
            lines = f.readlines()
            names = [i.strip() for i in lines]
        labels['actual_filename'] = names
    cols = labels.columns.to_list()
    cols = cols[-1:] + cols[-2:-1] + cols[:3]
    labels = labels[cols]
    return labels


def read_selected_images(images_folder_path, image_list_to_load, height, width, as_gray=True):
    from tqdm import tqdm
    from skimage import io, img_as_ubyte
    X = np.empty((len(image_list_to_load), height * width), dtype=np.ubyte)
    for i in tqdm(range(len(image_list_to_load))):
        a = io.imread(images_folder_path +
                      f"{image_list_to_load[i]}.jpg", as_gray=as_gray)
        a = img_as_ubyte(a)
        X[i, :] = a.reshape(1, -1)
    return X


def read_images_from_npy(npy_data_path, image_list_to_load):
    X = np.load(npy_data_path)
    X = X[image_list_to_load]
    return X


def get_decomposition_variance_score(decomposition_fitted, data_set, score_method=metrics.explained_variance_score):
    prediction = decomposition_fitted.inverse_transform(
        decomposition_fitted.transform(data_set))
    return score_method(data_set, prediction)


def get_decomposition_reconstruction_error_score(decomposition_fitted, data_set):
    prediction = decomposition_fitted.inverse_transform(
        decomposition_fitted.transform(data_set))
    return lng.norm(data_set-prediction)**2 / lng.norm(data_set)**2


def get_masked_reconstruction_error(decomposition_fitted, data_set, masked_data_set, mask_matrix):
    reconstructed_data_set = decomposition_fitted.inverse_transform(
        decomposition_fitted.transform(masked_data_set))
    M = (np.ones(mask_matrix.shape) - mask_matrix)
    dif = np.matmul(M, (data_set-reconstructed_data_set))
    return lng.norm(dif)**2/lng.norm(np.matmul(M, (data_set)))**2


def classification_error(y_test, y_hat):
    n_test = len(y_test)
    return np.sum(y_hat != y_test)/n_test


def break_labels(labels):
    '''
    Get the labels for the faces data set broken down to gender, race and age (bin) data sets
    argument labels is a pandas dataframe
    '''
    genders = ['male', 'female']
    races = ['white', 'black', 'asian', 'indian', 'other']
    ages = ['young', 'youngish', 'middle', 'old']
    labels_str_returned = {}
    labels_tuple_returned = {}
    i = 1
    for gender_i in labels.gender.unique():
        labels_gender = labels.loc[labels.gender == gender_i]
        for race_i in labels.race.unique():
            labels_gender_race = labels_gender.loc[labels_gender.race == race_i]
            for age_bin_i in labels.age_bin.unique():
                labels_gender_race_age_bin = labels_gender_race.loc[
                    labels_gender_race.age_bin == age_bin_i]
                labels_str_returned[f"{genders[gender_i]}_{races[race_i]}_{ages[age_bin_i]}"] = labels_gender_race_age_bin
                labels_tuple_returned[gender_i, race_i,
                                      age_bin_i] = labels_gender_race_age_bin
    return labels_str_returned, labels_tuple_returned


def get_vector_from_img(img_number, faces_path, labels):
    test_number = img_number
    if test_number >= labels.shape[0]:
        test_number = labels.shape[0]-1
    im_number = labels.loc[test_number].image_no
    a = io.imread(faces_path+f'{im_number}.jpg', as_gray=True)
    a = img_as_ubyte(a).reshape(1, -1)
    return a


def get_image_labels(image_number, labels):
    number = image_number
    if number >= labels.shape[0]:
        number = labels.shape[0]-1
    image_number = labels.loc[number].image_no
    gender = labels.loc[number].gender
    age_bin = labels.loc[number].age_bin
    race = labels.loc[number].race
    return (gender, race, age_bin, image_number)
