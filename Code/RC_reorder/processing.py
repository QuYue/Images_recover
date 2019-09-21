# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/12 19:41
@Author  : QuYue
@File    : processing.py
@Software: PyCharm
Introduction: Processings for images or matrices.
"""
#%% Import Packages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
#%% Functions
def rgb2gray(rgb,power=[0.229,0.587,0.114]):
    # get the gray picture from the RGB picture
    # Gray = R*0.299 + G*0.587 + B*0.114 (default)
    return np.dot(rgb, power)

def shuffle(image, axis=0, seed=None):
    """
    Shuffle one axis of the image.
    Parameters
    ---------
        image: ndarray
            The image which need to be shuffled.
        axis: int
            The axis, which need to be shuffled, of the image (default:0).
        seed: float
            The random seed (default:None).
    Returns
    ---------
        new_image: ndarray
            The image which has been shuffled.
        index: list
            The new order in which the axis is disrupted.
    """
    # shuffle the image
    np.random.seed(seed=seed) # set the random seed
    index = np.arange(image.shape[axis])
    np.random.shuffle(index)
    # axis change
    image = image.swapaxes(axis, 0) # change the choosen axis to the first axis
    new_image = image[index, ...]   # shuffle the choosen axis
    new_image = new_image.swapaxes(axis, 0) # change the axis back
    return new_image, index

def shift(image, axis=0, dis=1):
    """
    Shift one axis of the image.
    Parameters
    ---------
        image: ndarray
            The image which need to be shifted.
        axis: int
            The axis, which need to be shifted, of the image (default:0).
        dis: int
            The distance which image would be shifted. It can be negative. (default:1)
    Returns
    ---------
        new_image: ndarray
            The image which has been shifted.
    """
    # shift the image
    L = image.shape[axis]
    index0 = list(range(L))
    index = index0[dis:] + index0[0:dis]
    # axis change
    image = image.swapaxes(axis, 0) # change the choosen axis to the first axis
    new_image = image[index, ...] # shift the choosen axis
    new_image = new_image.swapaxes(axis, 0) # change the axis back
    return new_image


#%% Main Function
if __name__ == '__main__':
    im_name = 'YaoMing.jpg'
    im_path = './images/'
    image = mpimg.imread(im_path + im_name) # read the images
    plt.subplot(2,1,1)
    plt.imshow(image)
    plt.title(im_name)
    plt.axis('off')
    # gray
    image_gray = rgb2gray(image)
    plt.subplot(2, 3, 4)
    plt.imshow(image_gray, cmap='gray')
    plt.title('gray')
    plt.axis('off')
    # shuffle
    image_shuffled, _ = shuffle(image, axis=0)
    plt.subplot(2, 3, 5)
    plt.imshow(image_shuffled)
    plt.title('shuffled')
    plt.axis('off')
    plt.show()
    # shift
    dis = 100
    image_shifted = shift(image, axis=0, dis=100)
    plt.subplot(2,3,6)
    plt.imshow(image_shifted)
    plt.title('shifted %s' %dis)
    plt.axis('off')
    plt.show()

