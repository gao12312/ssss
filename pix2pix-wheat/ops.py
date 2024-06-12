import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
       

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def cbam_module(inputs,reduction_ratio=0.5,name=""):
    #  if reuse:
    #             tf.get_variable_scope().reuse_variables()
    #         else:
    #             assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("cbam_"+name, reuse=False):           #tf.AUTO_REUSE True 
        #假如输入是[batsize,h,w,channel]，
        #channel attension 因为要得到batsize * 1 * 1 * channel，它的全连接层第一层
        #隐藏层单元个数是channel / r, 第二层是channel，所以这里把channel赋值给hidden_num
        batch_size,hidden_num=inputs.get_shape().as_list()[0],inputs.get_shape().as_list()[3]
        #通道attension
        #全局最大池化，窗口大小为h * w，所以对于这个数据[batsize,h,w,channel]，他其实是求每个h * w面积的最大值
        #这里实现是先对h这个维度求最大值，然后对w这个维度求最大值，平均池化也一样
        maxpool_channel=tf.reduce_max(tf.reduce_max(inputs,axis=1,keep_dims=True),axis=2,keep_dims=True)
        avgpool_channel=tf.reduce_mean(tf.reduce_mean(inputs,axis=1,keep_dims=True),axis=2,keep_dims=True)
        # print(maxpool_channel)
        # print(avgpool_channel)
        #上面全局池化结果为batsize * 1 * 1 * channel，它这个拉平输入到全连接层
        #这个拉平，它会保留batsize，所以结果是[batsize,channel]
        #maxpool_channel = tf.layers.Flatten()(maxpool_channel)
        #avgpool_channel = tf.layers.Flatten()(avgpool_channel)
        maxpool_channel=tf.contrib.layers.flatten(maxpool_channel)
        avgpool_channel=tf.contrib.layers.flatten(avgpool_channel)
        # print(maxpool_channel)
        # print(avgpool_channel)
        #将上面拉平后结果输入到全连接层，第一个全连接层hiddensize = channel/r = channel * reduction_ratio，
        #第二哥全连接层hiddensize = channel
        mlp_1_max=tf.layers.dense(inputs=maxpool_channel,units=int(hidden_num*reduction_ratio),name="mlp_1",reuse=None,activation=tf.nn.relu)
        mlp_2_max=tf.layers.dense(inputs=mlp_1_max,units=hidden_num,name="mlp_2",reuse=None)
        #全连接层输出结果为[batsize,channel]，这里又降它转回到原来维度batsize * 1 * 1 * channel，
        mlp_2_max=tf.reshape(mlp_2_max,[batch_size,1,1,hidden_num])
        mlp_1_avg=tf.layers.dense(inputs=avgpool_channel,units=int(hidden_num*reduction_ratio),name="mlp_1",reuse=True,activation=tf.nn.relu)
        mlp_2_avg=tf.layers.dense(inputs=mlp_1_avg,units=hidden_num,name="mlp_2",reuse=True)
        mlp_2_avg=tf.reshape(mlp_2_avg,[batch_size,1,1,hidden_num])
        #将平均和最大池化的结果维度都是[batch_size,1,1,channel]相加，然后进行sigmod，维度不变
        channel_attention=tf.nn.sigmoid(mlp_2_max+mlp_2_avg)
        #和最开始的inputs相乘，相当于[batch_size,1,1,channel] * [batch_size,h,w,channel]
        #只有维度一样才能相乘,这里相乘相当于给每个通道作用了不同的权重
        channel_refined_feature=inputs*channel_attention
        # print(channel_attention)
        #print(channel_refined_feature)
        #空间attension
        #上面得到的结果维度依然是[batch_size,h,w,channel]，
        #下面要进行全局通道池化，其实就是一条通道里面那个通道的值最大，其实就是对channel这个维度求最大值
        #每个通道池化相当于将通道压缩到了1维，有两个池化，结果为两个[batch_size,h,w,1]feature map
        maxpool_spatial=tf.reduce_max(inputs,axis=3,keep_dims=True)
        avgpool_spatial=tf.reduce_mean(inputs,axis=3,keep_dims=True)

        #将两个[batch_size,h,w,1]的feature map进行通道合并得到[batch_size,h,w,2]的feature map
        max_avg_pool_spatial=tf.concat([maxpool_spatial,avgpool_spatial],axis=3)
    #然后对上面的feature map用1个7*7的卷积核进行卷积得到[batch_size,h,w,1]的feature map，因为是用一个卷积核卷的
    #所以将2个输入通道压缩到了1个输出通道
        conv_layer=tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same", activation=None)
        #然后再对上面得到的[batch_size,h,w,1]feature map进行sigmod，这里为什么要用一个卷积核压缩到1个通道，相当于只得到了一个面积的值
        #然后进行sigmod，因为我们要求的就是feature map面积上不同位置像素的中重要性，所以它压缩到了一个通道，然后求sigmod
        spatial_attention=tf.nn.sigmoid(conv_layer)
        #上面得到了空间attension feature map [batch_size,h,w,1]，然后再用这个和经过空间attension作用的结果相乘得到最终的结果
        #这个结果就是经过通道和空间attension共同作用的结果
        # print(spatial_attention)
        refined_feature=channel_refined_feature*spatial_attention
        #print(refined_feature)
    return refined_feature
