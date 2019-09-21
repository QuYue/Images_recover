# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/16 4:46
@Author  : QuYue
@File    : demo.py
@Software: PyCharm
Introduction: demo for RGB picture recover
"""
# %% Import Packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import sys
sys.path.append('..') # add the path which includes the packages
import RC_reorder.processing as processing
import RC_reorder.model as model
import RC_reorder.score as score

# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='A demo for RGB image recover.')
parser.add_argument('-i', '--image', type=str, default='lena.jpg', metavar='str',
                    help="the name of the image. (default: 'lena.jpg' )")
parser.add_argument('-p', '--path', type=str, default='../../Data/Images/', metavar='str',
                    help="the path of the picture. (default: '../../Data/Images/')")
Args = parser.parse_args() # the Arguments

#%% Main Function
if __name__ == '__main__':
    # %% Read Data
    image_name = Args.image # get the images name
    image_path = Args.path # get the path of the images
    image = mpimg.imread(image_path + image_name) # read the images
    # change the axis
    image_f = image.transpose(2, 0, 1)
    image_f = image_f[np.newaxis, ...]
    # show
    plt.ion()
    plt.figure(1) # show
    plt.subplot(1, 2, 1)
    fluency = score.Fluency_score(image_f)
    plt.imshow(image_f[0].transpose(1, 2, 0))
    plt.axis('off')
    plt.title('%s %d' %(image_name, fluency))

    # %% Shuffle
    shuffle_image, index0  = processing.shuffle(image_f, axis=2)
    index0 = np.array(index0) # change to ndarray
    plt.subplot(1, 2, 2) # show
    plt.imshow(shuffle_image[0].transpose(1, 2, 0))
    fluency = score.Fluency_score(shuffle_image)
    plt.title('shuffle %d' %fluency)
    plt.axis('off')
    plt.draw()

    # %% SVD image sort
    recover = model.ImageReorder(shuffle_image)
    new_image1, index = recover.svd_imsort()
    # show
    plt.figure(2)
    plt.imshow(new_image1[0].transpose(1, 2, 0))
    fluency = score.Fluency_score(new_image1)
    k_coeff, k_distance = score.Kendall_score(index0[index], range(len(index)))
    plt.title('SVD_imsort %d %.2f' %(fluency, k_coeff))
    plt.axis('off')
    plt.draw()

    # %% SVD greed
    plt.figure(3)
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        new_image2, index = recover.svd_greed(u_num=i)
        plt.imshow(new_image2[0].transpose(1, 2, 0))
        fluency = score.Fluency_score(new_image2)
        k_coeff, k_distance = score.Kendall_score(index0[index], range(len(index)))
        plt.title('u=%d %d %.2f' %(i, fluency, k_coeff))
        plt.axis('off')
    plt.draw()

    # %% Direct greed
    plt.figure(4)
    new_image3, index = recover.direct_greed()
    plt.imshow(new_image3[0].transpose(1, 2, 0))
    fluency = score.Fluency_score(new_image3)
    k_coeff, k_distance = score.Kendall_score(index0[index], range(len(index)))
    plt.title('Direct_greed %d %.2f' %(fluency,k_coeff))
    plt.axis('off')
    plt.draw()

    # # %% Symmetric
    # Score = []
    # L = new_image3.shape[2]
    # for i in range(L):
    #     new_image_shifted = processing.shift(new_image3, 2, i)
    #     symmetric, _ = score.Symmetric_score(new_image_shifted)
    #     Score.append(symmetric)
    # Score = np.array(Score)
    # best_symmetric = Score.argmax()
    # plt.figure(5)
    # plt.subplot(2,1,1)
    # new_image_shifted = processing.shift(new_image3, 2, best_symmetric)
    # plt.imshow(new_image_shifted[0].transpose(1, 2, 0))
    # plt.title('Symmetric')
    # plt.axis('off')
    # plt.subplot(2,1,2)
    # plt.plot(Score)
    # plt.vlines(best_symmetric,Score.min(), Score.max(),colors='red')
    # plt.axis('on')
    # plt.draw()

    # %%
    plt.ioff()
    plt.show()
