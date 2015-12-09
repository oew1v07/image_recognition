"""Image recognition algorithm written for COMP6223 at University of Southampton

This script takes training data to create a classifier that then tries to classify
a number of untrained images."""

from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from os.path import join, split #, exists, splitext

from skimage.feature import daisy
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import view_as_windows

from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC

import sys
from tqdm import tqdm
from traceback import print_exc


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
image_classes_names = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4,
                       'industrial': 5, 'Insidecity': 6, 'kitchen': 7, 'livingroom': 8,
                       'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
                       'Street':13, 'Suburb': 14, 'TallBuilding':15}

image_classes_int = {}

for k, v in image_classes_names.items():
    image_classes_int[v] = k

image_folders = list(image_classes_names.keys())


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


def create_tinys_array(folder, export=False, pixels=16):
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

    return array, list_of_files


def write_output(glob_list, y, run_no):
    """Writes the output in the format specified

    Parameters
    ----------
    glob_list: list
        The glob list should be in the full path format.
    y: ndarray
        The predicted classes in integer format.

    Raises
    ------
    ValueError
        If the length of glob list is not the same as y.
    """
    if len(glob_list) != len(y):
        raise ValueError("List of globs and y are not the same length")

    else:

        list_of_jpgs = []
        classes = []

        for i in glob_list:
            [path, jpg] = split(i)
            list_of_jpgs.append(jpg)

        for i in y:
            im_class = image_classes_int[i]
            classes.append(im_class)

        out_path = join(path, 'run{}.txt'.format(run_no))

        with open(out_path, 'w') as t:
            for i in range(len(glob_list)):
                line = list_of_jpgs[i] + " " + classes[i].lower()
                t.write(line)
                t.write("\n")
    return out_path


def training_tiny_image(tr_folder='/Users/robin/COMP6223/cw3/training', pixels=16,
                        export=False):
    print(image_folders, tr_folder)
    paths = [join(tr_folder, i) for i in image_folders]

    length = len(paths)

    #Using the fact that there are 100 in each folder
    tiny_images = np.zeros((length*100, np.square(pixels)))
    targets = np.zeros((length*100,1))

    start = 0
    chunk = 100
    for i in paths:
        [c, d] = split(i)
        class_num = image_classes_names[d]
        #no minus 1 as numpy arrays are exclusive at end.
        end_point = start + chunk
        array, list_of_files = create_tinys_array(i, export, pixels)
        # Put array into the big array for use in sklearn
        tiny_images[start:end_point, :] = array
        targets[start:end_point, :] = class_num * np.ones((100, 1))
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
    y = y.ravel()
    neigh = KNeighborsClassifier(n_neighbors)
    neigh.fit(X,y)

    # gets mean accuracy on the data
    acc = neigh.score(X, y)
    return neigh, acc


def split_test_knn(X, y, n_neighbors=5, test_size=0.2, run_num=100, run_no=1):
    tr_acc = []
    tst_acc = []

    for i in range(run_num):
        [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=test_size)

        [neigh, acc_tr] = KNN(X_train, y_train, n_neighbors)

        # Put training accuracy into dataframe
        tr_acc.append(acc_tr)

        # Calculate the test accuracy
        acc_tst = neigh.score(X_test, y_test)
        tst_acc.append(acc_tst)

    return tr_acc, tst_acc, neigh


def run1(tr_folder='/Users/robin/COMP6223/cw3/training',
         test_folder='/Users/robin/COMP6223/cw3/testing',
         n_neighbors=[5], pixels=16, export=False, run_num=100, test_size=0.2):

    run_no = 1
    # Training the algorithm
    X, y = training_tiny_image(tr_folder, pixels, export)

    ma_trs = []
    ma_tsts = []

    # Doing a cross validation
    for i in n_neighbors:
        tr_acc, tst_acc, neigh = split_test_knn(X, np.ravel(y), n_neighbors=i,
                                                run_num=run_num, run_no=run_no,
                                                test_size=test_size)
        # find mse for training and test data
        ma_tr = np.mean(tr_acc)
        ma_tst = np.mean(tst_acc)

        ma_trs.append(ma_tr)
        ma_tsts.append(ma_tst)
        np.save('run{}_tr_acc_k_{}.npy'.format(run_no, i), tr_acc)
        np.save('run{}_tst_acc_k_{}.npy'.format(run_no, i), tst_acc)

    # Calculating the optimum k where optimum is the max test accuracy
    opt_k = n_neighbors[np.argmax(ma_tsts)]

    plt.figure()
    plt.plot(n_neighbors, ma_trs, n_neighbors, ma_tsts)
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.savefig('run{}_k_vs_acc.png'.format(run_no))

    [neigh, a] = KNN(X, y, n_neighbors=opt_k)

    # Now we've found the optimum k we shall try on the test data
    test_array, list_of_files = create_tinys_array(test_folder)

    test_out = neigh.predict(test_array)

    write_output(list_of_files, test_out, run_no)

    return ma_trs, ma_tsts, n_neighbors, test_out


def get_dense_patches(image, patch_size = 8, sample_rate = 4):
    """Gets dense patches of pixels from an image.

    This uses the view_as_windows function from scikit-image.

    Parameters
    ----------
    image: ndarray
        Image to find dense patches within.
    patch_size: int (default: 8)
        Size of each patch. 8 x 8 patches are recommended.
    sample_rate: int (default: 4)
        How many pixels apart each patch should be chosen in both x and y
        directions.

    Returns
    -------
    out: ndarray
        An array of which the rows make up a ravel of each dense patch.
    """

    window_shape = (patch_size, patch_size)
    patches = view_as_windows(image, window_shape = window_shape,
                                    step = sample_rate)

    shp = patches.shape
    # Resize the array so that each row is one dense patch.
    out = np.reshape(patches, (shp[0]*shp[1], shp[2]*shp[3]))

    # Normalize the data
    m = out.mean(axis = 1)

    # Have to transpose as it won't take subtract along columns properly
    out = (np.transpose(out) - m).transpose()

    # Make unit length along the rows
    out = normalize(out, axis = 1)

    return out


def get_dense_patches_for_folder(folder, patch_size = 8, sample_rate = 4):
    """Creates one array of dense patches for every image in a folder.

    Parameters
    ----------
    folder: String
        folder with all jpgs
    patch_size: int (default: 8)
        Size of each patch. 8 x 8 patches are recommended.
    sample_rate: int (default: 4)
        How many pixels apart each patch should be chosen in both x and y
        directions.

    Returns
    -------
    list_of_jpgs: list(String)
        ['0.jpg', '1.jpg', ..., '99.jpg']
    la_patches_of_each_image: list(ndarray)
        [array_patches(0.jpg), array_patches(1.jpg), ...]
    a_patches_for_class: ndarray
        array_patches(bedroom)
    """

    # Get a list of all the jpegs in a folder
    pattern = join(folder,'*.jpg')

    list_of_files = glob(pattern)

    list_of_jpgs = []

    # Create empty list for patches to go in. Each item is an array for each
    # image
    la_patches_of_each_image = []

    # load in each jpg separately, create dense patch array and add to an
    # empty list

    for im in list_of_files:
        # Create a String list of files e.g. ['0.jpg','1.jpg', ...]
        [c, d] = split(im)
        list_of_jpgs.append(d)
        # import image into an array
        image = image_to_array(im)
        # create array of patches for each image
        patches = get_dense_patches(image, patch_size, sample_rate)
        # append each array to list of images
        la_patches_of_each_image.append(patches)
        print('Finished image at {}'.format(datetime.now().time()))

    # join all the arrays in the list using np.vstack,
    # but we also need to have each individual image as a list ofor the future
    # k nearest neighbor. the vstack is needed for sampling and the list
    # is needed for the nearest neighbour after clustering.
    a_patches_for_class = np.vstack(la_patches_of_each_image)

    return list_of_jpgs, la_patches_of_each_image, a_patches_for_class


def get_dense_patches_for_all_classes(tr_folder = '/Users/robin/COMP6223/cw3/training',
                                      patch_size = 8, sample_rate = 4):
    """Creates a matrix of all features for each class

    Parameters
    ----------
    tr_folder: String (default: '/Users/robin/COMP6223/cw3/training')
        A filepath to the folder containing all the other class folders.
    patch_size: int (default: 8)
        Size of each patch. 8 x 8 patches are recommended.
    sample_rate: int (default: 4)
        How many pixels apart each patch should be chosen in both x and y
        directions.

    Returns
    -------
    lla_patches_of_each_image: list(list(ndarray))
        lla_patches_of_each_image[0] = [array_patches(0.jpg), array_patches(1.jpg), ...]
        where class is determined by order of class.

    ll_list_of_jpgs: list(list(String))
        ll_list_of_jpgs[0] = ['0.jpg', '1.jpg', ..., '99.jpg']
        but this is in the order they were originally loaded.

    ll_list_of_files: list(list(String))
        ll_list_of_files[0] = ['/Users/robin/COMP6223/cw3/training/bedroom/0.jpg', ...]
        in the order they were originally loaded.

    la_patches_for_class: list(ndarray)
        la_patches_for_class[0] = [array_patches(bedroom), array_patches(coast), ...]
        where class is determined by order of class.

    order_of_classes: list(int)
        order_of_classes in order it was loaded. i.e. [15,2,3,5, ...]
    """

    # Each class path
    paths = [join(tr_folder, i) for i in image_folders]

    # At the lowest level an array of patches for each image
    lla_patches_of_each_image = []

    # At the lowest level strings of '0.jpg'
    ll_list_of_jpgs = []

    # At the lowest level an array of patches for each class (vstack of all images) - for sampling!
    la_patches_for_class = []

    order_of_classes = []

    for path in paths:

        [list_of_jpgs,
         la_patches_of_each_image,
         a_patches_for_class] = get_dense_patches_for_folder(path,
                                                             patch_size,
                                                             sample_rate)

        lla_patches_of_each_image.append(la_patches_of_each_image)

        ll_list_of_jpgs.append(list_of_jpgs)

        la_patches_for_class.append(a_patches_for_class)

        # Create an order of classes
        [c, d] = split(path)
        class_num = image_classes_names[d]
        order_of_classes.append(class_num)
        print('Finished {} at {}'.format(d, datetime.now().time()))

    return [lla_patches_of_each_image, ll_list_of_jpgs,
            la_patches_for_class, order_of_classes]


def sample_patches(order_of_classes, la_patches_for_class, sample_num = 500):

    # At the lowest level an array of patches from each class: 15 x 500 row array
    la_list_of_samples = []

    # Samples are for each class
    for index, i in enumerate(order_of_classes):

        # Get the corresponding patch from la_patches_for_class (an array)
        a_patches_for_class = la_patches_for_class[index]

        # Create an array of sample_num random integers between 0 and
        # len(a_patches_for_class)
        sample_index = np.random.randint(len(a_patches_for_class),size = sample_num)

        #Sample from the array using the indexes
        sample = a_patches_for_class[sample_index,:]

        # Append to the list
        la_list_of_samples.append(sample)

    return la_list_of_samples


def find_clusters(la_list_of_samples, order_of_classes, cluster_num = 50):
    """Creates the codebook to do linear classification on.

    This creates cluster_num x centres for each image class.

    Parameters
    ----------
    la_list_of_samples: list(ndarray)
        A list of all the ndarrays with the samples as rows in the ndarray.
        Each item represents a class
    order_of_classes: list(int)
        A list of the order the classes were loaded in
    cluster_num: int
        k value or number of clusters for each image class to have

    Returns
    -------
    la_list_of_centres: list(ndarray)
        A list with each item being the array of cluster_num x centres for
        each class.
    la_list_of_words: list(ndarray)
        Each item is an array of targets for each centre.
    """

    la_list_of_centres = []
    la_list_of_words = []

    for index, i in enumerate(la_list_of_samples):
        X = i
        # Remember this is for each of the class sets - and that there's no such
        # thing as accuracy for this type of clustering. So they'll be
        # cluster_num x clusters for each class.
        kmc = KMeans(n_clusters = cluster_num)
        kmc.fit(X)
        centres = kmc.cluster_centers_
        words = order_of_classes[index]*np.ones((cluster_num,1))

        la_list_of_centres.append(centres)
        la_list_of_words.append(words)

    return la_list_of_centres, la_list_of_words


def find_histograms_for_images(neigh, patches_of_each_image, order_of_classes):
    """ Creates histogram for an array of patches. This is for one image

    Parameters
    ----------
    neigh: KNeighborsClassifier
        The classifier with which to classify each patch
    patches_of_each_image: list(ndarray)
        This needs to be an array for each image with each row in that array
        being a "patch".

    Returns
    -------
    histogram: ndarray
        A (1,15) array with each entry corresponding to the number of patches
        identified as each class given by the index. ie an image may have 10
        patches and may have 7 patches identified by the KNN classifier as being
        class 10 but also 1 each for classes 1, 2 and 3. This would result in an
        ndarray like so: [1,1,1,0,0,0,0,0,0,0,7,0,0,0,0] (confusing because the
        first class is the 0th index).
    """

    # Now we have an array of each images patches.
    # Take each patch from an image and find its nearest cluster using
    # the KNN.
    predicted_classes_of_patches = neigh.predict(patches_of_each_image)

    # Sum each seperate values - like histogram without the graph!
    # This is for each image
    predicted_hist, bin_edges = np.histogram(predicted_classes_of_patches,
                                              bins=list(range(1, len(order_of_classes) + 2)))

    return predicted_hist


def get_training_data_for_histogram(la_list_of_centres, la_list_of_words,
                                    lla_patches_of_each_image, order_of_classes,
                                    run_no):
    """Puts training data into a format so the histogram function can be run

    Parameters
    ----------

    Returns
    -------
    targets: ndarray
        The class with which each image is associated
    """

    # Train a knn classifier using all the centres and the targets
    # Take all the centres and words and stack them
    # a_centres is the X
    a_centres = np.vstack(la_list_of_centres)

    # a_words is the y
    a_words = np.vstack(la_list_of_words)

    neigh, acc = KNN(a_centres, a_words, n_neighbors = 1)

    la_list_of_histograms = []
    la_list_of_targets = []

    # For each class in lla_patches_of_each_image - i.e. 15
    for index, i in tqdm(enumerate(lla_patches_of_each_image)):

        # Get each images class by getting index of i in
        # lla_patches_of_each_image and then taking that element of
        # order_of_classes.
        class_num = order_of_classes[index]

        for j in tqdm(i):
            histogram = find_histograms_for_images(neigh, j, order_of_classes)
            la_list_of_targets.append(class_num)
            # Append to respective lists
            la_list_of_histograms.append(histogram)

        print('Saved for %d' % order_of_classes[index])
        np.save('run{}_{}.npy'.format(run_no, index), histogram)

    # Vstack these lists! These are the inputs to the linear classifier
    # We'll need to calculate these for the test and training data.
    # This is the X for the linear classifier
    histograms = np.vstack(la_list_of_histograms)

    # This is the y for the linear classifier
    targets = np.vstack(la_list_of_targets)

    print('About to save')
    np.save('run{}_histograms.npy', histograms)
    np.save('run{}_targets.npy', targets)

    return histograms, targets, neigh


def one_vs_all(X, y, test_size=0.4, run_num=4, svm_type='linear'):
    """Trains 15 1 vs all SVM linear classifiers"""
    # Python has a wonderful wrapper function that creates 1 vs all classifiers!
    if type == 'linear':
        estimator = LinearSVC()
    else:
        # This will automatically use RBF functions
        estimator = SVC()

    ovr = OneVsRestClassifier(estimator=estimator)

    acc_tr = []
    acc_tst = []

    for i in range(run_num):
        [X_train, X_test, y_train, y_test] = train_test_split(X, y,
                                                              test_size=test_size)
        # Train the classifier
        ovr.fit(X_train, y_train.ravel())

        # Work out the score on the training data. However there is nothing
        # to optimise for - we are just getting an idea of the accuracy for
        # training vs test data. box plot opportunity!
        tr_acc = ovr.score(X_train, y_train.ravel())
        tst_acc = ovr.score(X_test, y_test.ravel())

        acc_tr.append(tr_acc)
        acc_tst.append(tst_acc)

    # Put all training items in here for the final ovr
    ovr = OneVsRestClassifier(estimator=LinearSVC())
    ovr.fit(X, y.ravel())

    full_tr_acc = ovr.score(X, y.ravel())

    return ovr, acc_tr, acc_tst, full_tr_acc


def run2_train(sample_num=2000, cluster_num=200, test_size=0.2, run_num=100,
               patch_size=8, sample_rate=4):

    print('This started at {}'.format(datetime.now().time()))
    [lla_patches_of_each_image, ll_list_of_jpgs,
     la_patches_for_class,
     order_of_classes] = get_dense_patches_for_all_classes(patch_size=patch_size,
                                                           sample_rate=sample_rate)
    print('Ended at {}'.format(datetime.now().time()))

    np.save('run2_order_of_classes.npy', order_of_classes)

    # Sampling
    la_list_of_samples = sample_patches(order_of_classes, la_patches_for_class,
                                        sample_num=sample_num)

    # Creating a codebook
    la_list_of_centres, la_list_of_words = find_clusters(la_list_of_samples,
                                                         order_of_classes,
                                                         cluster_num=cluster_num)

    [histograms, targets,
     neigh] = get_training_data_for_histogram(la_list_of_centres,
                                              la_list_of_words,
                                              lla_patches_of_each_image,
                                              order_of_classes,
                                              run_no=2)

    joblib.dump(neigh, 'run2_neigh.pkl')

    ovr, acc_tr, acc_tst, full_tr_acc = one_vs_all(histograms, targets,
                                                   test_size=test_size,
                                                   run_num=run_num,
                                                   svm_type='linear')

    # Do box plots of accuracies training vs test

    joblib.dump(ovr, 'run2_ovr.pkl')

    return neigh, ovr, order_of_classes


def run2_test(neigh=None, ovr=None, order_of_classes=None,
              test_folder='/Users/robin/COMP6223/cw3/testing', test_size=0.2,
              run_num=100):

    if neigh is None:
        path = join('/Users/robin/COMP6223/cw3/', 'run2_neigh.pkl')
        neigh = joblib.load(path)
    if ovr is None:
        path = join('/Users/robin/COMP6223/cw3/', 'run2_ovr.pkl')
        ovr = joblib.load(path)
    if order_of_classes is None:
        path = join('/Users/robin/COMP6223/cw3/', 'run2_order_of_classes.npy')
        order_of_classes = np.load(path)

    # THIS IS NOW DONE - DON'T NEED TO DO IT AGAIN
    # To get the accuracy we need histograms and targets

    histograms = np.load(join('/Users/robin/COMP6223/cw3/', 'run2_histograms.npy'))
    targets = np.load(join('/Users/robin/COMP6223/cw3/', 'run2_targets.npy'))

    ovr, acc_tr, acc_tst, full_tr_acc = one_vs_all(histograms, targets,
                                                   test_size=test_size,
                                                   run_num=run_num,
                                                   svm_type='linear')

    # Do box plots of the accuracy.
    acc_tr = np.array(acc_tr)
    acc_tst = np.array(acc_tst)

    # Saving accuracies
    np.save('run2_acc_tr', acc_tr)
    np.save('run2_acc_tst', acc_tst)
    np.save('run2_full_tr_acc', full_tr_acc)

    data = [acc_tr, acc_tst]
    fig,ax1 = plt.subplots(figsize=(8,6))
    bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlabel('Type of Error')
    ax1.set_ylabel('Accuracy')

    xticknames = plt.setp(ax1, xticklabels=['Training', 'Test'])
    plt.setp(xticknames, fontsize=14)
    plt.savefig('ovr_accuracy.png')

    # Creating test data
    [test_list_of_jpgs,
     test_la_patches_of_each_image,
     test_a_patches_for_class] = get_dense_patches_for_folder(test_folder,
                                                              patch_size=8,
                                                              sample_rate=4)

    # List for each of the test histograms
    test_list_of_histograms = []

    # Take the test data and work out it histogram
    for index, i in enumerate(test_la_patches_of_each_image):
        predicted_hist = find_histograms_for_images(neigh, i, order_of_classes)
        test_list_of_histograms.append(predicted_hist)
        print("Finished element number {}".format(index))

    print("Putting list together as array")
    test_histograms = np.vstack(test_list_of_histograms)

    # SAVE HERE!
    print('About to save')
    np.save('run2_test_histograms.npy', test_histograms)

    predicted_class = ovr.predict(test_histograms)

    write_output(test_list_of_jpgs, predicted_class, run_no=2)

    return predicted_class


def DAISY_extractor(image, step=4, radius=15, rings=3, num_histograms=6,
                    orientations=8, visualize=False, normalization='daisy'):
    """Calculates daisy descriptors and puts each in one row of an array.

    The step size needs to be quite small to get as many descriptors for the
    number to be as large as the dense patches.

    Parameters
    ----------
    step: int (default: 4)
        Distance between each sampling point
    radius: in (default: 15)
        radius of outermost ring in pixels
    rings: int (default: 3)
        Number of rings around sampling point
    num_histograms: int (default: 6)
        Number of "petals" in each ring
    orientations: int (default: 6)
        Number of orientations (directions) per histogram
    visualize: bool (default: False)
        If true creates a visualisation of the daisy on the image given
    normalization: ['l1', 'l2', 'daisy', 'off'], (default: 'daisy')
        What type of normalization to use. 'daisy' does the l2 norm for each
        histogram

    Returns
    -------
    out: ndarray
        An array with each row corresponding to a daisy featurevector, which
        has been normalized along the rows. For items created with the same
        histograms, orientations and rings the width of this array will be
        constant at (rings * histograms + 1) * orientations
    """

    if visualize:
        descs, descs_img = daisy(image, step=step, radius=radius, rings=rings,
                                 histograms=num_histograms, orientations=orientations,
                                 visualize=visualize, normalization=normalization)

        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(descs_img)
        name = 'daisy_step{}_radius{}_rings{}.png'.format(step, radius, rings)
        plt.savefig(name)
    else:
        descs = daisy(image, step=step, radius=radius, rings=rings,
                      histograms=num_histograms, orientations=orientations,
                      visualize=visualize, normalization=normalization)

    # reshapes the daisy outputs so each pixels histogram is a row in an array.
    # takes the "band" dimension and puts it at the front. shp should be of the
    # format 40 x 40 x 152 (where the 40s are the spatial part of the image,
    # making the 152 the histogram of each pixel, which we would like ravelled)
    shp = descs.shape
    descs_shaped = np.rollaxis(descs, 2)

    # It has to be done like this otherwise the wrong elements will be in the
    # wrong places and then transpose it.
    out = descs_shaped.reshape((shp[2], shp[0]*shp[1])).T

    # Normalize the data
    m = out.mean(axis = 1)

    # Have to transpose as it won't take subtract along columns properly
    out = (np.transpose(out) - m).transpose()

    # Make unit length along the rows
    out = normalize(out, axis = 1)

    return out


def get_daisy_descs_for_folder(folder, step, radius, rings, num_histograms,
                               orientations, visualize, normalization):
    """Creates one array of daisy descriptors for every image in a folder.

    Parameters
    ----------
    folder: String
        folder with all jpgs
    step: int (default: 4)
        Distance between each sampling point
    radius: in (default: 15)
        radius of outermost ring in pixels
    rings: int (default: 3)
        Number of rings around sampling point
    histograms: int (default: 6)
        Number of "petals" in each ring
    orientations: int (default: 6)
        Number of orientations (directions) per histogram
    visualize: bool (default: False)
        If true creates a visualisation of the daisy on the image given
    normalization: ['l1', 'l2', 'daisy', 'off'], (default: 'daisy')
        What type of normalization to use. 'daisy' does the l2 norm for each
        histogram

    Returns
    -------
    list_of_jpgs: list(String)
        ['0.jpg', '1.jpg', ..., '99.jpg']
    la_patches_of_each_image: list(ndarray)
        [array_patches(0.jpg), array_patches(1.jpg), ...]
    a_patches_for_class: ndarray
        array_patches(bedroom)

    Start: 23:29:16 End: 23:29:56 0.5s to do one image!
    """

    # Get a list of all the jpegs in a folder
    pattern = join(folder,'*.jpg')

    list_of_files = glob(pattern)

    list_of_jpgs = []

    # Create empty list for patches to go in. Each item is an array for each
    # image
    la_daisy_of_each_image = []

    # load in each jpg separately, create dense daisy descriptor array and add
    # to an empty list

    for im in list_of_files:
        # Create a String list of files e.g. ['0.jpg','1.jpg', ...]
        [c, d] = split(im)
        list_of_jpgs.append(d)
        # import image into an array
        image = image_to_array(im)
        # create array of patches for each image
        daisy = DAISY_extractor(image, step=step, radius=radius, rings=rings,
                                num_histograms=num_histograms,
                                orientations=orientations, visualize=visualize,
                                normalization=normalization)
        # append each array to list of images
        la_daisy_of_each_image.append(daisy)
        print('Finished image at {}'.format(datetime.now().time()))

    # join all the arrays in the list using np.vstack,
    # but we also need to have each individual image as a list ofor the future
    # k nearest neighbor. the vstack is needed for sampling and the list
    # is needed for the nearest neighbour after clustering.
    a_daisy_for_class = np.vstack(la_daisy_of_each_image)

    return list_of_jpgs, la_daisy_of_each_image, a_daisy_for_class


def get_daisy_descs_for_all_classes(step, radius, rings, num_histograms, orientations,
                                    visualize, normalization,
                                    tr_folder = '/Users/robin/COMP6223/cw3/training'):
    """Creates a matrix of all features for each class

    Parameters
    ----------
    tr_folder: String (default: '/Users/robin/COMP6223/cw3/training')
        A filepath to the folder containing all the other class folders.
    step: int (default: 4)
        Distance between each sampling point
    radius: in (default: 15)
        radius of outermost ring in pixels
    rings: int (default: 3)
        Number of rings around sampling point
    num_histograms: int (default: 6)
        Number of "petals" in each ring
    orientations: int (default: 6)
        Number of orientations (directions) per histogram
    visualize: bool (default: False)
        If true creates a visualisation of the daisy on the image given
    normalization: ['l1', 'l2', 'daisy', 'off'], (default: 'daisy')
        What type of normalization to use. 'daisy' does the l2 norm for each
        histogram

    Returns
    -------
    lla_daisy_of_each_image: list(list(ndarray))
        lla_patches_of_each_image[0] = [array_patches(0.jpg), array_patches(1.jpg), ...]
        where class is determined by order of class.

    ll_list_of_jpgs: list(list(String))
        ll_list_of_jpgs[0] = ['0.jpg', '1.jpg', ..., '99.jpg']
        but this is in the order they were originally loaded.

    la_daisy_for_class: list(ndarray)
        la_patches_for_class[0] = [array_patches(bedroom), array_patches(coast), ...]
        where class is determined by order of class.

    order_of_classes: list(int)
        order_of_classes in order it was loaded. i.e. [15,2,3,5, ...]
    """

    # Each class path
    paths = [join(tr_folder, i) for i in image_folders]

    # At the lowest level an array of patches for each image
    lla_daisy_of_each_image = []

    # At the lowest level strings of '0.jpg'
    ll_list_of_jpgs = []

    # At the lowest level an array of patches for each class (vstack of all images) - for sampling!
    la_daisy_for_class = []

    order_of_classes = []

    for path in paths:

        [list_of_jpgs,
         la_daisy_of_each_image,
         a_daisy_for_class] = get_daisy_descs_for_folder(path, step, radius,
                                                         rings, num_histograms,
                                                         orientations, visualize,
                                                         normalization)

        lla_daisy_of_each_image.append(la_daisy_of_each_image)

        ll_list_of_jpgs.append(list_of_jpgs)

        la_daisy_for_class.append(a_daisy_for_class)

        # Create an order of classes
        [c, d] = split(path)
        class_num = image_classes_names[d]
        order_of_classes.append(class_num)
        print('Finished {} at {}'.format(d, datetime.now().time()))

    return [lla_daisy_of_each_image, ll_list_of_jpgs,
            la_daisy_for_class, order_of_classes]


def run3_train(sample_num=2000, cluster_num=200, test_size=0.4, run_num=100,
               step=4, radius=15, rings=3, num_histograms=6, orientations=8,
               visualize=False, normalization='daisy'):
    """Runs the daisy descriptor for all training images and calculates accuracy."""

    run_no = 3

    print('This started at {}'.format(datetime.now().time()))
    [lla_daisy_of_each_image, ll_list_of_jpgs,
     la_daisy_for_class,
     order_of_classes] = get_daisy_descs_for_all_classes(step=step, radius=radius,
                   rings=rings, num_histograms=num_histograms, orientations=orientations,
                   visualize=visualize, normalization=normalization)
    print('Ended at {}'.format(datetime.now().time()))

    np.save('run3_order_of_classes.npy', order_of_classes)

    # From here on the code for run2 and run1 is almost identical!
    # Sampling
    la_list_of_samples = sample_patches(order_of_classes, la_daisy_for_class,
                                        sample_num=sample_num)

    # Creating a codebook
    la_list_of_centres, la_list_of_words = find_clusters(la_list_of_samples,
                                                         order_of_classes,
                                                         cluster_num=cluster_num)

    [histograms, targets,
     neigh] = get_training_data_for_histogram(la_list_of_centres,
                                              la_list_of_words,
                                              lla_daisy_of_each_image,
                                              order_of_classes,
                                              run_no=3)

    joblib.dump(neigh, 'run3_neigh.pkl')

    ovr, acc_tr, acc_tst = one_vs_all(histograms, targets,
                                      test_size=test_size, run_num=run_num,
                                      svm_type='non-linear')


    joblib.dump(ovr, 'run3_ovr.pkl')

    return neigh, ovr, order_of_classes


def run3_test(neigh=None, ovr=None, order_of_classes=None, step=4, radius=15,
              rings=3, num_histograms=6, orientations=8, visualize=False,
              normalization='daisy', test_folder='/Users/robin/COMP6223/cw3/testing',
              test_size=0.2, run_num=100):
    """Runs the daisy descriptor run for all test images."""

    run_no = 3

    if neigh is None:
        path = join('/Users/robin/COMP6223/cw3/', 'run3_neigh.pkl')
        neigh = joblib.load(path)
    if ovr is None:
        path = join('/Users/robin/COMP6223/cw3/', 'run3_ovr.pkl')
        ovr = joblib.load(path)
    if order_of_classes is None:
        path = join('/Users/robin/COMP6223/cw3/', 'run3_order_of_classes.npy')
        order_of_classes = np.load(path)

    # THIS IS NOW DONE - DON'T NEED TO DO IT AGAIN
    # To get the accuracy we need histograms and targets

    histograms = np.load(join('/Users/robin/COMP6223/cw3/', 'run3_histograms.npy'))
    targets = np.load(join('/Users/robin/COMP6223/cw3/', 'run3_targets.npy'))

    ovr, acc_tr, acc_tst, full_tr_acc = one_vs_all(histograms, targets,
                                                   test_size=test_size,
                                                   run_num=run_num,
                                                   svm_type='non-linear')

    # Do box plots of the accuracy.
    acc_tr = np.array(acc_tr)
    acc_tst = np.array(acc_tst)

    # Saving accuracies
    np.save('run3_acc_tr.npy', acc_tr)
    np.save('run3_acc_tst.npy', acc_tst)
    np.save('run3_full_tr_acc.npy', full_tr_acc)

    data = [acc_tr, acc_tst]
    fig, ax1 = plt.subplots(figsize = (8,6))
    bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlabel('Type of Error')
    ax1.set_ylabel('Accuracy')

    xticknames = plt.setp(ax1, xticklabels=['Training', 'Test'])
    plt.setp(xticknames, fontsize=14)
    plt.savefig('ovr_accuracy.png')

    # Creating test data
    [test_list_of_jpgs,
     test_la_daisy_of_each_image,
     test_a_daisy_for_class] = get_daisy_descs_for_folder(test_folder,
                                                          step=step, radius=radius,
                                                          rings=rings,
                                                          num_histograms=num_histograms,
                                                          orientations=orientations,
                                                          visualize=visualize,
                                                          normalization=normalization)

    # List for each of the test histograms
    test_list_of_histograms = []

    # Take the test data and work out it histogram
    for index, i in enumerate(test_la_daisy_of_each_image):
        predicted_hist = find_histograms_for_images(neigh, i, order_of_classes)
        test_list_of_histograms.append(predicted_hist)
        print("Finished element number {}".format(index))

    print("Putting list together as array")
    test_histograms = np.vstack(test_list_of_histograms)

    # SAVE HERE!
    print('About to save')
    np.save('run{}_test_histograms.npy'.format(run_no), test_histograms)

    predicted_class = ovr.predict(test_histograms)

    write_output(test_list_of_jpgs, predicted_class, run_no=3)

    return predicted_class


def plot_accuracies(run1_tr_acc=None, run1_tst_acc=None, run2_tr_acc=None,
                    run2_tst_acc=None, run3_tr_acc=None, run3_tst_acc=None):
    pass


# if __name__ == '__main__':
#     # if the commange line has three arguments then the images have been
#     # provided
#     if len(sys.argv) == 3:
#         image1 = sys.argv[1]
#         image2 = sys.argv[2]
#         run_hybrid(image1, image2)
#     else:
#         print("Usage: python hybrid_image.py image1 image2")
#         sys.exit()
