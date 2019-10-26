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

# %% Functions
def show(image):
    if len(image.shape) == 4:
        plt.imshow(image[0].transpose(1, 2, 0))
    else:
        plt.imshow(image, cmap='gray')


# %% Get the arguments from cmd.
parser = argparse.ArgumentParser(description='A demo for RGB image recover.')
parser.add_argument('-i', '--image', type=str, default='lena.jpg', metavar='str',
                    help="the name of the image. (default: 'lena.jpg' )")
parser.add_argument('-p', '--path', type=str, default='../../Data/Images/', metavar='str',
                    help="the path of the picture. (default: '../../Data/Images/')")
parser.add_argument('-rc', '--RowColumn', action='store_true', default=False,
                    help="if reorder both row and column. (default: False)")
parser.add_argument('-g', '--gray', action='store_true', default=False,
                    help="if reorder the gray image. (default: False)")
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
    plt.suptitle('Task', y=0.9, size=20)
    if Args.gray:
        plt.subplot(1, 3, 1) # 灰度图要画三张
    else:
        plt.subplot(1, 2, 1)
    fluency = score.Fluency_score(image_f)
    show(image_f)
    plt.axis('off')
    plt.title('%s %d' %(image_name, fluency))

    # %% Gray
    if Args.gray:
        gray_image = processing.rgb2gray(image)
        plt.subplot(1, 3, 2) # show
        plt.imshow(gray_image,cmap='gray')
        fluency = score.Fluency_score(gray_image)
        plt.axis('off')
        plt.title('%s-Gray %d' %(image_name, fluency))

    # %% Shuffle
    if Args.RowColumn == False:
        if Args.gray: 
            shuffle_image, index0 = processing.shuffle(gray_image, axis=0)
        else:
            shuffle_image, index0  = processing.shuffle(image_f, axis=2)
        index0 = np.array(index0) # change to ndarray
    else:
        shuffle_image, index_r0 = processing.shuffle(image_f, axis=2)
        shuffle_image, index_c0 = processing.shuffle(shuffle_image, axis=3)

    if Args.gray:
        plt.subplot(1, 3, 3)
    else:
        plt.subplot(1, 2, 2)  # show
    show(shuffle_image)
    fluency = score.Fluency_score(shuffle_image)
    plt.title('shuffle %d' % fluency)
    plt.axis('off')
    plt.draw()

    #%% Recover
    if Args.RowColumn == False:
        # SVD image sort
        recover = model.ImageReorder(shuffle_image)
        new_image1, index = recover.svd_imsort()
        # show
        plt.figure(2)
        show(new_image1)
        fluency = score.Fluency_score(new_image1)
        k_coeff, k_distance = score.Kendall_score(index0[index], range(len(index)))
        plt.title('SVD_imsort %d %.2f' %(fluency, k_coeff))
        plt.axis('off')
        plt.draw()

        # SVD greed
        plt.figure(3)
        for i in range(1, 21):
            plt.subplot(4, 5, i)
            new_image2, index = recover.svd_greed(u_num=i)
            show(new_image2)
            fluency = score.Fluency_score(new_image2)
            k_coeff, k_distance = score.Kendall_score(index0[index], range(len(index)))
            plt.title('u=%d %d %.2f' %(i, fluency, k_coeff))
            plt.axis('off')
        plt.draw()

        # Direct greed
        plt.figure(4)
        new_image3, index = recover.direct_greed()
        show(new_image3)
        fluency = score.Fluency_score(new_image3)
        k_coeff, k_distance = score.Kendall_score(index0[index], range(len(index)))
        plt.title('Direct_greed %d %.2f' %(fluency,k_coeff))
        plt.axis('off')
        plt.draw()
    else:
        plt.figure(2)
        plt.suptitle('Reorder Row and Column')
        temp_image = shuffle_image
        method = []
        method.append('Direct_gread')
        method.append('SVD_imsort')
        for i in range(1,30): method.append('SVD_greed%s' %i)
        for i in range(5):
            methods = []
            fluencys = []
            for j in range(2):
                recover = model.ImageReorder(temp_image)
                t_image = []
                fluency=[]
                new_image, index = recover.direct_greed()
                t_image.append(new_image)
                fluency.append(score.Fluency_score(t_image[-1]))
                for k in range(30):
                    new_image, index = recover.svd_greed(k+1)
                    t_image.append(new_image)
                    fluency.append(score.Fluency_score(t_image[-1]))
                pos = np.array(fluency).argmin()
                temp_image = t_image[pos]
                temp_image = temp_image.swapaxes(2, 3)
                plt.subplot(2, 5, (i+1)+5*j)
                plt.title('greed %d' % (fluency[pos]))
                show(temp_image)
                methods.append(method[pos])
                fluencys.append(fluency[pos])
                plt.axis('off')
                plt.draw()
            print('step%s: method %s %s| fluency [%s %s]' %(i+1, methods[0], methods[1], fluencys[0], fluencys[1]))

    # %% Symmetric
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
