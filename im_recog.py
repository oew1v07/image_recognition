"""Image recognition algorithm written for COMP6223 at University of Southampton

This script takes training data to create a classifier that then tries to classify
a number of untrained images."""

import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
# from numpy.testing import assert_array_equal
from skimage.io import imread, imsave
from skimage.transform import resize
from glob import glob
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import cross_validation
# from scipy import fftpack, misc
# from scipy.ndimage.interpolation import zoom
# import matplotlib.pyplot as plt
from os.path import join, split #, exists, splitext
from os import getcwd

import sys

def _odd(number):
    """Raises an error if a number isn't odd"""
    if number % 2 == 0:
        raise TypeError("Only odd sizes are supported. "
                        "Got {number}.".format(number = number))

def _dim(number):
    """Raises an error if a number is greater than 2"""
    if number > 2:
        raise ValueError("Only dimensions of 2 are allowed. "
                        "Got {number}.".format(number = number))
    elif number < 2:
        raise ValueError("Only dimensions of 2 are allowed. "
                        "Got {number}.".format(number = number))

def _check_type(ar, data_types):
    """Raises an error if ar is not of the type given"""
    if not type(ar) in data_types:
        raise TypeError("Only {} types are supported. Got {}.".format(data_types,
                                                                      type(ar)))
# Create a dict for easy look up
image_classes = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4,
                 'industrial': 5, 'Insidecity': 6, 'kitchen': 7, 'livingroom': 8,
                 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
                 'Street':13, 'Suburb': 14, 'TallBuilding':15}

image_folders = list(image_classes.keys())

def image_to_array(image):
    """Reads in image and turns values into a numpy array

    Parameters
    ----------
    image: string (filepath to image)
        Image to be read in and put into array. It can be of shape (x,y,z)
        where x and y are any size and z is the number of bands (1 or 3)

    Raises
    ------
    OSError: cannot identify image file
        If the file given is not a readable image type

    Returns
    -------
    out: ndarray
        Image as an (M x N) or (M x N x 3) array
    """
    # Uses PIL plugin to read image in
    out = imread(image)
    return out

def create_tiny_image(image, pixels = 16):
    """Creates a tiny image which is a resized, squared version of the original.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to have tiny image made. It is expected that images will be of
        the shape (x,y,1) i.e. number of bands is 1
    pixels: int (default: 16)
        The number of pixels wide that the tiny image should be.

    Raises
    ------

    ValueError
        If the number of pixels is not an integer value
    ValueError
        If the number of dimensions of the image is not two.

    Returns
    -------
    out: ndarray
        The tiny image in a (1 x pixels^2) array
    """
    _dim(image.ndim)
    _check_type(pixels, [int])

    # Get dimensions of image and work out which is smaller
    y = image.shape[0]
    x = image.shape[1]

    if x == y:
        clipped = image
    else:
        # width is the width of the square for the image to be cropped to
        width = min(x, y)

        # using a boolean to times how much each x or y is less than width
        # if c is False then the x direction needs cropping
        # if c is True then the y direction needs cropping
        c = x < y

        # Find out how much either side we need to come in by.
        x_diff = (x - width)*(not c)
        y_diff = (y - width)*c

        # Clip the image
        if c:
            if y_diff % 2:
                left = np.floor(y_diff/2)
                right = np.ceil(y_diff/2)
            else:
                left = y_diff/2
                right = left

            clipped = image[left:-right, :]

        else:
            if x_diff % 2:
                left = np.floor(x_diff/2)
                right = np.ceil(x_diff/2)
            else:
                left = x_diff/2
                right = left

            clipped = image[:, left:-right]

    # Shape for tiny image to be
    output_shape = (pixels, pixels)

    # Resize the image
    tiny = resize(image, output_shape)

    # Creates a 1-d array of all the elements
    out = np.ravel(tiny)

    # Normalize the output
    out = normalize(out - out.mean())

    return tiny, out

def create_tinys_array(folder, export = False, pixels = 16):
    """Creates an array of tiny images from images within a specified folder.

    Parameters
    ----------
    folder: String
        Folder where the images to have tiny images made are stored
    export: bool (default: False)
        Whether a csv of the array created should be created in the folder.
    pixels: int (default: 16)
        The number of pixels wide that each tiny image should be.

    Raises
    ------

    ValueError
        If the number of pixels is not an integer value
    ValueError
        If the number of dimensions of the image is not two.

    Returns
    -------
    out: ndarray
        An array of all the tiny images in the specified folder.
        Size is (n x pixels^2) where n is number of images.
    """
    # Get a list of all the jpegs in a folder
    pattern = join(folder,'*.jpg')

    list_of_files = glob(pattern)

    # Create empty array for tinys to go in
    array = np.zeros((len(list_of_files),np.square(pixels)))

    # load in each jpg separately, create tiny image and add to an
    # empty dataframe

    for im in list_of_files:
        out = image_to_array(im)
        [tiny, row] = create_tiny_image(out, pixels)
        array[list_of_files.index(im),:] = row

    if export:
        [c, image_class] = split(folder)
        name = join(c, image_class + '_tiny_image.jpg')
        imsave(name, array)

    return array

def training_tiny_image(export = False, pixels = 16):
    paths = [join('/Users/olivia/COMP6223/cw3/training', i) for i in image_folders]

    length = len(paths)

    #Using the fact that there are 100 in each folder
    tiny_images = np.zeros((length*100, np.square(pixels)))
    targets = np.zeros((length*100,1))

    start = 0
    chunk = 100
    for i in paths:
        [c,d] = split(i)
        class_num = image_classes[d]
        #no minus 1 as numpy arrays are exclusive at end.
        end_point = start + chunk
        array = create_tinys_array(i, export, pixels)
        # Put array into the big array for use in sklearn
        tiny_images[start:end_point,:] = array
        targets[start:end_point,:] = class_num*np.ones((100,1))
        start = start + chunk
    return tiny_images, targets

def KNN(X, y, n_neighbors = 5):
    """Fits a K nearest neighbour classifier to the given data.

    Parameters
    ----------
    X: ndarray
        The data to fit the classifier to.
    y: ndarray
        The targets to try and predict.
    n_neighbors: int
        Number of neighbours each point should be compared to. Incorrect
        spelling for the sklearn method - so I don't get confused.

    Returns
    -------
    neigh: KNeighborsClassifier
        The classifier object to then do more predictions elsewhere
    acc: float
        The accuracy of the classifier on the training data

    """
    neigh = KNeighborsClassifier(n_neighbors)
    neigh.fit(X,y)

    # gets mean accuracy on the data
    acc = neigh.score(X, y)
    return neigh, acc

def split_test_knn(X, y, n_neighbors = 5, test_size = 0.4, run_num = 4):
    tr_acc = []
    tst_acc = []

    for i in range(run_num):
        [X_train, X_test, y_train, y_test] = cross_validation.train_test_split(X, y, test_size=test_size)

        [neigh, acc_tr] = KNN(X_train, y_train, n_neighbors)

        # Put training accuracy into dataframe
        tr_acc.append(acc_tr)

        pred = neigh.predict(X_test)

        print(pred[:10])

        #print(metrics.classification_report(y_test, pred))

        # Calculate the test accuracy
        acc_tst = neigh.score(X_test, y_test)
        tst_acc.append(acc_tst)

    return tr_acc, tst_acc

def run1(test_folder, pixels = 16, export = False):
    X, y = training_tiny_image(export, pixels)
    tr_acc, tst_acc = split_test_knn(X, np.ravel(y))

    return tr_acc, tst_acc

if __name__ == '__main__':
    # if the commange line has three arguments then the images have been
    # provided
    if len(sys.argv) == 3:
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        run_hybrid(image1, image2)
    else:
        print("Usage: python hybrid_image.py image1 image2")
        sys.exit()
