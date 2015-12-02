"""Image recognition algorithm written for COMP6223 at University of Southampton

This script takes training data to create a classifier that then tries to classify
a number of untrained images."""

import numpy as np
# from numpy.fft import fft2, ifft2, fftshift
# from numpy.testing import assert_array_equal
from skimage.io import imread, imsave
from skimage.transform import resize
# from scipy import fftpack, misc
# from scipy.ndimage.interpolation import zoom
# import matplotlib.pyplot as plt
# from os.path import split, splitext, join, exists
# from os import mkdir

import sys

def _odd(number):
    """Raises an error if a number isn't odd"""
    if number % 2 == 0:
        raise TypeError("Only odd sizes are supported. "
                        "Got {number}.".format(number = number))

def _dim(number):
    """Raises an error if a number is greater than 2"""
    if number > 2:
        raise TypeError("Only dimensions of 2 or fewer are allowed. "
                        "Got {number}.".format(number = number))

def _check_type(ar, data_type):
    """Raises an error if ar is not of the type given"""
    if not type(ar) == data_type:
        raise TypeError("Only {} types are supported. Got {}.".format(data_type,
                                                                      type(ar)))

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

    Returns
    -------
    out: ndarray
        The tiny image in a (1 x pixels^2) array
    """
    _dim(number)

    # Get dimensions of image and work out which is smaller
    y = image.shape[0]
    x = image.shape[1]

    # width is the width of the square for the image to be cropped to
    if y <= x:
        width = y
    else:
        width = x

    

    # Find the centroid of the array
    centre_x = int(np.floor(x/2))
    centre_y = int(np.floor(y/2))


    output_shape =

    resize(image, output_shape,  )

    out = zeros((1, pixels^2))


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
