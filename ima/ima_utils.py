import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt


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
    test_img = skimage.io.imread(faces_path+f"{image_no}.jpg", as_gray=as_gray)
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