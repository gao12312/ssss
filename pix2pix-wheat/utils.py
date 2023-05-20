# coding=utf-8
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import time
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

INPUT_C_DIM = 11
IMG_SIZE = 256
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def compare(s1, s2):
    s1 = map(int, s1.split('.')[1].split('/')[-1].split('-'))
    s2 = map(int, s2.split('.')[1].split('/')[-1].split('-'))
    if s1[0] != s2[0]:
        return -1 if s1[0] < s2[0] else 1
    else:
        if s1[1] != s2[1]:
            return -1 if s1[1] < s2[1] else 1
        else:
            if s1[2] != s2[2]:
                return -1 if s1[2] < s2[2] else 1
            else: return 0

def process_hyper_data(files_X, files_Y):
    """
    该函数可将X和Y合并为一个batch并返回
    x的每个样本有187张图片，首先需要将其合并为一个，然后再进行操作
    Args:
        files_X:x的一个batch的样本，但是里面存储的是字符串（文件名）[filename1,filename2, ..., filename_inputdim]
        files_Y:

    Returns:模型所能处理的batch

    """
    start_time = time.time()
    batch_size = len(files_Y)
    #print('batch is :',batch_size)
    x_processed = process_X_hyper(files_X, batch_size)
    # x_processed .shape is : (batchsize, 256, 256, 187)
    y_origin = np.array([imread_hyper(i) for i in files_Y])
    # 进行拼接(1, 256, 256, 187) + (1, 256, 256, 3) --> (1, 256, 256, 190)
    #print("x_processed.shape is :", x_processed.shape)
    #print("y shape is :", y_origin.shape)
    res_A, res_B = [], []
    for i in range(batch_size):
        x_temp, y_temp = hyper_preprocess_A_and_B(x_processed[i], y_origin[i], flip=True)
        x_temp = x_temp / 127.5 - 1.0
        y_temp = y_temp / 127.5 - 1.0
        res_A.append(x_temp)
        res_B.append(y_temp)

    res_B = np.expand_dims(np.array(res_B), -1)
    res_A = np.array(res_A)
    #print("res_A, res_B shape is :", res_A.shape, res_B.shape)
    batch = np.concatenate((res_A, res_B), axis=3)
    #print(batch.shape)
    return batch

def hyper_preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = img_A[550-320:]
        img_B = img_B[550-320:]
        # print("img_A type is :", type(img_A))
        img_A1 = img_A[:, :, 0]
        img_A1 = scipy.misc.imresize(img_A1, [fine_size, fine_size])
        img_A1 = np.expand_dims(img_A1, -1)
        for i in range(1, INPUT_C_DIM):
            split = img_A[:, :, i]
            split = scipy.misc.imresize(split, [fine_size, fine_size])
            split = np.expand_dims(split, -1)
            img_A1 = np.concatenate((img_A1, split), axis=-1)
        img_A = img_A1
        # img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        # print('here')
        # print("img_A shape is :", img_A.shape)
        # print("img_B shape is :", img_B.shape)

    return img_A, img_B

def process_X_hyper(files_X, batch_size, is_grayscale=True):

    '''
    Args:
        files_X:[fn11, fn12, fn13, fn21, fn22, fn23, ..., fnN1, fnN2, fnN3] (N*3)
        batch_size: N
    Returns:[N, h, w, input_dim]
    '''
    start_time = time.time()
    x_origin = np.array([imread_hyper(i) for i in files_X])
    #print(x_origin.shape)
    # 注意这里不能直接使用reshape，是因为该函数的底层逻辑是直接flatten然后在重构的，但是这样并不能达到我们的预期
    x_processed = []
    for eve_batch in range(batch_size):
        # x_t = np.expand_dims(x_origin[eve_batch * INPUT_C_DIM], axis=-1)  # (1024, 543)
        x_t = [np.expand_dims(x_origin[i], -1) for i in range(eve_batch * INPUT_C_DIM, (eve_batch+1) * INPUT_C_DIM)]
        x_t = np.concatenate(x_t, -1)
        # for i in range(1, INPUT_C_DIM):
        #     x_t = np.concatenate((x_t, np.expand_dims(x_origin[eve_batch * INPUT_C_DIM + i], axis=-1)), axis=-1)
        # if is_grayscale:
        #     x_t = x_t[2092-1004:]
        #     x_t = scipy.misc.imresize(x_t, [256, 256])
        x_processed.append(x_t)
    x_processed = np.array(x_processed)
    return x_processed


def imread_hyper(impath):
    """
    专门读取高光谱图像的函数，并且对读取的图像进行处理，将大小转化为256X256
    Args:
        impath: 图像文件路径
        isgrayscale: 是否为灰度图像，默认为否

    Returns: （256,256，1）或者（256,256,3）
    """
    image = imread(impath)
    # return scipy.misc.imresize(image, [IMG_SIZE, IMG_SIZE])
    return image

def imread(path, is_grayscale = True):  #huidu picture
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def inverse_transform(images):
    return (images+1.)/2.

def save_images_hyper(images, size, image_path, is_gray=True):
    if is_gray:
        h, w = images.shape[1], images.shape[2]
        image = np.zeros((h*size[0], w*size[1]))
        for idx in range(size[0]):
            image[idx*h: (idx+1)*h, :] = images[idx, :, :, 0]
        scipy.misc.toimage(image).save(image_path)


#
# def transform(image, npx=64, is_crop=True, resize_w=64):
#     # npx : # of pixels width/height of image
#     if is_crop:
#         cropped_image = center_crop(image, npx, resize_w=resize_w)
#     else:
#         cropped_image = image
#     return np.array(cropped_image)/127.5 - 1.



# def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
#     if is_test:
#         img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
#         img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
#     else:
#
#         img_A = scipy.misc.imresize(img_A, [load_size, load_size])
#         img_B = scipy.misc.imresize(img_B, [load_size, load_size])
#
#         h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
#         w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
#         img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]   # 0.02, load_size-fine_size
#         img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]
#
#
#         if flip and np.random.random() > 0.5:
#             img_A = np.fliplr(img_A)
#             img_B = np.fliplr(img_B)
#         # print("img_A shape is :", img_A.shape)
#         # print("img_B shape is :", img_B.shape)
#
#     return img_A, img_B
#
# # -----------------------------
#
# def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
#     return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)
#

# '''

# import functools
# from glob import glob
# dataset_name='hyperspectra'
# #process_hyper_data('./datasets/{}/train/X/*.jpg','./datasets/{}/train/Y/*.jpg')
# #process_hyper_data('/home/lzy/mjx/pix2pix-tensorflow-master/datasets/{}/train/X/*.jpg','/home/lzy/mjx/pix2pix-tensorflow-master/datasets/{}/train/Y/*.jpg')
# data_X = glob('./datasets/{}/train/X/*.jpg'.format(dataset_name))
# X_files = np.array(sorted(data_X, key=functools.cmp_to_key(compare)))
# #print(X_files)
# data_Y = glob('./datasets/{}/train/Y/*.jpg'.format(dataset_name))
# Y_files = np.array(sorted(data_Y, key=functools.cmp_to_key(compare)))
# process_hyper_data(X_files,Y_files)
# '''