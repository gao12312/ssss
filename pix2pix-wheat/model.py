# coding=utf-8
from __future__ import division

import functools
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from ops import *
from utils import *
import scipy.misc


class pix2pix(object):
    def __init__(self, sess, batch_size,
                 image_size,output_size,
                 input_c_dim, output_c_dim,hyper_dim,
                 sample_size=1,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 dataset_name='', checkpoint_dir=None, sample_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size                #输入图像大小
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim           #输入输出维度
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        self.hyper_dims = hyper_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.Image_usage = np.array(range(0,self.input_c_dim))   #选择的图像，全部波段
        #self.Image_usage = (np.linspace(89, 99, 11)).astype(int)    
        #print(self.Image_usage)
        self.data_nums = len(glob('./datasets/{}/train/Y/*.jpg'.format(self.dataset_name)))   #标签Y中样本数量
        print(self.data_nums)
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        # generate left from right
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]     #A 光谱图像
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim+self.output_c_dim]  #B 合并后的图像
        self.fake_B = self.generator(self.real_A)    #生成的光谱图像
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        print("real_A is :", self.real_A.shape)
        print("real_B is :", self.real_B.shape)
        print("fake_B is :", self.fake_B.shape)
        print("real_AB is :", self.real_AB)
        print("fake_AB is :", self.fake_AB)
        #原patchGAN判别器
        # self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        # self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)
        #多尺度判别器
        self.D, self.D_logits = self.multiscale_discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.multiscale_discriminator(self.fake_AB, reuse=True)
              
        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                      + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)          #生成器损失与判别器损失
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()
        # self.saver.save(self.sess, './checkpoint/facades_'+ str(self.batch_size) + '_' + str(self.output_size)+ '.ckpt')

    def load_random_samples_hyper(self,idx):
        """unique for hyperspectra data"""
        data_X = glob('./datasets/{}/train/X/*.jpg'.format(self.dataset_name))           #x file
        X_files = np.array(sorted(data_X, key=functools.cmp_to_key(compare)))
        #print(len(X_files))
        data_Y = glob('./datasets/{}/train/Y/*.jpg'.format(self.dataset_name))            #y file
        Y_files = np.array(sorted(data_Y, key=functools.cmp_to_key(compare)))
        random_file_Numbers=[idx]                #当前训练样本保存
        #random_file_Numbers = np.random.choice(range(0, self.data_nums), self.batch_size)  # 随机取到batch_size个样本，可自定义选择样本数
        #print(random_file_Numbers)
        file_names_X_sample = np.array(
            [X_files[self.hyper_dims * j + self.Image_usage-1] for j in random_file_Numbers]).flatten()
        file_names_Y_sample = np.array([Y_files[num] for num in random_file_Numbers])
        return process_hyper_data(file_names_X_sample, file_names_Y_sample,self.input_c_dim,self.image_size)

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples_hyper(idx)
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        # print("samples.shape is :", samples.shape)
        save_images_hyper(samples, [self.batch_size, 1],
                          './{}/train_{:02d}_{:03d}.jpg'.format(sample_dir, epoch, idx))  #保存过程sample，需要修改文件名为random_file_Numbers
        #print("[Sample] d_loss: {:.2f}, g_loss: {:.2f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
       
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
                                       self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        
        history_dloss=[]
        history_gloss=[]
        for epoch in xrange(args.epoch):
            data_X = glob('./datasets/{}/train/X/*.jpg'.format(self.dataset_name))    #排序获得x_file下的所有样本
            data_X = np.array(sorted(data_X, key=functools.cmp_to_key(compare)))
            data_Y = glob('./datasets/{}/train/Y/*.jpg'.format(self.dataset_name))         #排序获得y_file下的所有样本
            data_Y = np.array(sorted(data_Y, key=functools.cmp_to_key(compare)))  
            batch_idxs = min(len(data_Y), args.train_size) // self.batch_size
            print("banch_dixs is :", batch_idxs)
            
            # 一次训练epoch需要更新所有的批次batch_idxs,保存每次的loss
            epoch_dloss=0
            epoch_gloss=0
            for idx in xrange(0, batch_idxs):  #idx=step
                '''
                这里需要对代码做的更改有：
                源代码是直接读取一张图片然后将其处理为x和y放入batch中
                -->分别读取x和y保存到两个变量中然后进行concatenate操作
                '''
                '''
                batch_files = data_X[idx*self.input_c_dim*self.batch_size:(idx+1)*self.input_c_dim*self.batch_size]
                batch_files = data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                # 这一步操作带来的结果： 
                # 在通道那一维 0：output_c_dim存储y，output_c_dim存储x
                batch = [load_data(batch_file) for batch_file in batch_files]
                '''
                # 更改代码块
                # batch_files_X = data_X[idx * self.input_c_dim * self.batch_size: (idx + 1) * self.input_c_dim * self.batch_size]
                batch_files_X = np.array(
                   [data_X[idx * self.batch_size * self.hyper_dims + eve_batch * self.hyper_dims + self.Image_usage]    #定位每个idx的batchsize对应Image_usage选择的所有样本
                    for eve_batch in range(self.batch_size)]).flatten()
                batch_files_Y = data_Y[idx * self.batch_size: (idx + 1) * self.batch_size]
                #print(batch_files_X)
                #print(batch_files_Y)
                batch = process_hyper_data(batch_files_X, batch_files_Y,self.input_c_dim,self.image_size)
                # 更改代码块
                #print ("batch is :", (np.array(batch)).shape)

                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                #print("batch_images shape is :", batch_images.shape)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                epoch_dloss +=errD_fake+errD_real
                epoch_gloss +=errG
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.2f, g_loss: %.2f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                #if np.mod(counter, 10) == 1:       # 10步随机保存训练数据到sample
                #    self.sample_model(args.sample_dir, epoch, idx)
                
                if epoch == args.epoch-1:              #只保存最后一个epoch的样本
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter,batch_idxs) == 0:         #batch_idxs步保存checkpoint
                    self.save(args.epoch,args.checkpoint_dir, counter)
            
            epoch_dloss=epoch_dloss/batch_idxs    #每个样本的平均loss
            epoch_gloss=epoch_gloss/batch_idxs
            print(epoch_dloss)
            print(epoch_gloss)
            history_dloss.append(epoch_dloss)
            history_gloss.append(epoch_gloss)
        with open("./loss314-unet++-discriminator.txt","a") as f:
            f.write("g_loss")
            [f.write(str(i)+" ") for i in history_gloss]
            f.write("\n")
            f.write("d_loss")
            [f.write(str(i)+" ") for i in history_dloss]
            
# '''            
#     def discriminator(self, image, y=None, reuse=False):

#         with tf.variable_scope("discriminator") as scope:

#             # image is 256 x 256 x (input_c_dim + output_c_dim)
#             if reuse:
#                 tf.get_variable_scope().reuse_variables()
#             else:
#                 assert tf.get_variable_scope().reuse == False

#             h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
#             # h0 is (128 x 128 x self.df_dim)
#             h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
#             # h1 is (64 x 64 x self.df_dim*2)
#             h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
#             # h2 is (32x 32 x self.df_dim*4)
#             h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')))
#             # h3 is (16 x 16 x self.df_dim*8)
#             h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
#             return tf.nn.sigmoid(h4), h4
# '''

    def multiscale_discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            # Create and run discriminators at multiple scales
            scales = [1, 2, 4]  # Define the scales you want to use
            discriminator_outputs = []
            for scale in scales:
                scope_name = "scale_%d" % scale  # 使用字符串格式化创建唯一的名称
                with tf.variable_scope(scope_name):
                    w =image.shape[1]//scale 
                    h =image.shape[2]//scale
                    # 计算缩放后的高度和宽度
                    new_height = tf.cast(h ,tf.int32)
                    new_width = tf.cast(w ,tf.int32)
                    scaled_image = tf.image.resize_images(image, [new_width, new_height])
                    #print(scaled_image.shape)
                    # Resize the image to the current scale
                    #scaled_image = resize_image(image, scale)  # Implement 'resize_image' as needed
                    # Build the discriminator for the current scale
                    h0 = lrelu(conv2d(scaled_image, self.df_dim, name='d_h0_conv'))
                    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                    h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')))
                    h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                    discriminator_outputs.append(h4)

            # Calculate the average discriminator output
            average_output = tf.reduce_mean(discriminator_outputs, axis=0)
            # Return both the individual and average outputs
            return tf.nn.sigmoid(average_output),h4

    '''
    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128, s256 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128),int(s / 256)
            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e7_resized = tf.image.resize_images(e7,[s256,s256])  # Resize e7 to match e6 size
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)
            #cbam_e8=cbam_module(e8)
            #添加CBAM,添加残差连接ResBlock_CBAM
            #e8=e8+cbam_e8
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim * 8], name='g_d1',
                                                     with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)
            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim * 8], name='g_d2',
                                                     with_w=True)
            
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2,e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)
            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim * 8], name='g_d3',
                                                     with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim * 8], name='g_d4',
                                                     with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim * 4], name='g_d5',
                                                     with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)
            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim * 2], name='g_d6',
                                                     with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)
            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim], name='g_d8',
                                                     with_w=True)
            #print(self.d8.shape)
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(self.d8)
    '''
    def generator(self, image, y=None):
            with tf.variable_scope("generator") as scope:
                s = self.output_size
                s2, s4, s8, s16, s32, s64, s128, s256 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                    s / 64), int(s / 128),int(s / 256)
                # image is (256 x 256 x input_c_dim)
                e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
                e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
                # e3 is (32 x 32 x self.gf_dim*4)
                e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
                # e4 is (16 x 16 x self.gf_dim*8)
                e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
                # e5 is (8 x 8 x self.gf_dim*8)
                e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'))
                # e6 is (4 x 4 x self.gf_dim*8)
                e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'))
                # e7 is (2 x 2 x self.gf_dim*8)
                e7_resized = tf.image.resize_images(e7,[s256,s256])  # Resize e7 to match e6 size
                e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv'))
                # e8 is (1 x 1 x self.gf_dim*8)
                #添加CBAM
                cbam_e8=cbam_module(e8)
                e8=e8+cbam_e8
                
                #-------------------残差连接-------------
                self.ds8, self.ds8_w, self.ds8_b = deconv2d(tf.nn.relu(e8),
                                                        [self.batch_size, s128, s128, self.gf_dim * 8], name='g_ds8',
                                                        with_w=True)
                residual_connection_e7= tf.add(self.ds8, e7)  
                # print("e7 and residual_connection_e7:")
                # print(e7.shape)
                # print(residual_connection_e7.shape)
                
                self.ds7, self.ds7_w, self.ds7_b = deconv2d(tf.nn.relu(e7),
                                                        [self.batch_size, s64, s64, self.gf_dim * 8], name='g_ds7',
                                                        with_w=True)
                residual_connection_e6= tf.add(self.ds7, e6)  
                # print("e6 and residual_connection_e6:")
                # print(e6.shape)
                # print(residual_connection_e6.shape)

                self.ds6, self.ds6_w, self.ds6_b = deconv2d(tf.nn.relu(e6),
                                                        [self.batch_size, s32, s32, self.gf_dim * 8], name='g_ds6',
                                                        with_w=True)
                residual_connection_e5= tf.add(self.ds6, e5)
                #print(e5.shape)
                #print(residual_connection_e5.shape)
                
                self.ds5, self.ds5_w, self.ds5_b = deconv2d(tf.nn.relu(e5),
                                                        [self.batch_size, s16, s16, self.gf_dim * 8], name='g_ds5',
                                                        with_w=True)
                residual_connection_e4 = tf.add(self.ds5, e4)
                # print("e4 and residual_connection_e4:")
                # print(e4.shape)
                # print(residual_connection_e4.shape)
                
                self.ds4, self.ds4_w, self.ds4_b = deconv2d(tf.nn.relu(e4),
                                                        [self.batch_size,s8,s8, self.gf_dim * 4], name='g_ds4',
                                                        with_w=True)
                residual_connection_e3 = tf.add(self.ds4,e3)
                # print("e3 and residual_connection_e3:")
                # print(e3.shape)
                # print(residual_connection_e3.shape)
                
                self.ds3, self.ds3_w, self.ds3_b = deconv2d(tf.nn.relu(e3),
                                                        [self.batch_size,s4,s4, self.gf_dim * 2], name='g_ds3',
                                                        with_w=True)
                residual_connection_e2 = tf.add(self.ds3,e2)
                # print("e2 and residual_connection_e2:")
                # print(e2.shape)
                # print(residual_connection_e2.shape)
                
                self.ds2, self.ds2_w, self.ds2_b = deconv2d(tf.nn.relu(e2),
                                                        [self.batch_size,s2,s2, self.gf_dim], name='g_ds2',
                                                        with_w=True)
                residual_connection_e1 = tf.add(self.ds2,e1)
                # print("e1 and residual_connection_e1:")
                # print(e1.shape)
                # print(residual_connection_e1.shape)
                #-------------------残差连接-------------
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                        [self.batch_size, s128, s128, self.gf_dim * 8], name='g_d1',
                                                        with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, residual_connection_e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)
                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                        [self.batch_size, s64, s64, self.gf_dim * 8], name='g_d2',
                                                        with_w=True)
                
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2,residual_connection_e6], 3)
                #print(d2.shape)
                # d2 is (4 x 4 x self.gf_dim*8*2)
                #------------------------------
                
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                        [self.batch_size, s32, s32, self.gf_dim * 8], name='g_d3',
                                                        with_w=True)
                d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, residual_connection_e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)

                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                        [self.batch_size, s16, s16, self.gf_dim * 8], name='g_d4',
                                                        with_w=True)
                d4 = self.g_bn_d4(self.d4)
                d4 = tf.concat([d4, residual_connection_e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                        [self.batch_size, s8, s8, self.gf_dim * 4], name='g_d5',
                                                        with_w=True)
                d5 = self.g_bn_d5(self.d5)
                d5 = tf.concat([d5,residual_connection_e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)
                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                        [self.batch_size, s4, s4, self.gf_dim * 2], name='g_d6',
                                                        with_w=True)
                d6 = self.g_bn_d6(self.d6)
                d6 = tf.concat([d6, residual_connection_e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)
                self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                                                        [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
                d7 = self.g_bn_d7(self.d7)
                d7 = tf.concat([d7, residual_connection_e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                                                        [self.batch_size, s, s, self.output_c_dim], name='g_d8',
                                                        with_w=True)
                #print(self.d8.shape)
                # d8 is (256 x 256 x output_c_dim)
                return tf.nn.tanh(self.d8)
     
    def sampler(self, image, y=None):
            with tf.variable_scope("generator") as scope:
                scope.reuse_variables()
                s = self.output_size
                s2, s4, s8, s16, s32, s64, s128, s256 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                    s / 64), int(s / 128),int(s / 256)
                # image is (256 x 256 x input_c_dim)
                e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
                e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
                # e3 is (32 x 32 x self.gf_dim*4)
                e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
                # e4 is (16 x 16 x self.gf_dim*8)
                e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
                # e5 is (8 x 8 x self.gf_dim*8)
                e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'))
                # e6 is (4 x 4 x self.gf_dim*8)
                e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'))
                # e7 is (2 x 2 x self.gf_dim*8)
                e7_resized = tf.image.resize_images(e7,[s256,s256])  # Resize e7 to match e6 size
                e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv'))
                # e8 is (1 x 1 x self.gf_dim*8)
                #添加CBAM
                cbam_e8=cbam_module(e8)
                e8=e8+cbam_e8
                #-------------------残差连接-------------
                self.ds8, self.ds8_w, self.ds8_b = deconv2d(tf.nn.relu(e8),
                                                        [self.batch_size, s128, s128, self.gf_dim * 8], name='g_ds8',
                                                        with_w=True)
                residual_connection_e7= tf.add(self.ds8, e7)  
                # print("e7 and residual_connection_e7:")
                # print(e7.shape)
                # print(residual_connection_e7.shape)
                
                self.ds7, self.ds7_w, self.ds7_b = deconv2d(tf.nn.relu(e7),
                                                        [self.batch_size, s64, s64, self.gf_dim * 8], name='g_ds7',
                                                        with_w=True)
                residual_connection_e6= tf.add(self.ds7, e6)  
                # print("e6 and residual_connection_e6:")
                # print(e6.shape)
                # print(residual_connection_e6.shape)

                self.ds6, self.ds6_w, self.ds6_b = deconv2d(tf.nn.relu(e6),
                                                        [self.batch_size, s32, s32, self.gf_dim * 8], name='g_ds6',
                                                        with_w=True)
                residual_connection_e5= tf.add(self.ds6, e5)
                #print(e5.shape)
                #print(residual_connection_e5.shape)
                
                self.ds5, self.ds5_w, self.ds5_b = deconv2d(tf.nn.relu(e5),
                                                        [self.batch_size, s16, s16, self.gf_dim * 8], name='g_ds5',
                                                        with_w=True)
                residual_connection_e4 = tf.add(self.ds5, e4)
                # print("e4 and residual_connection_e4:")
                # print(e4.shape)
                # print(residual_connection_e4.shape)
                
                self.ds4, self.ds4_w, self.ds4_b = deconv2d(tf.nn.relu(e4),
                                                        [self.batch_size,s8,s8, self.gf_dim * 4], name='g_ds4',
                                                        with_w=True)
                residual_connection_e3 = tf.add(self.ds4,e3)
                # print("e3 and residual_connection_e3:")
                # print(e3.shape)
                # print(residual_connection_e3.shape)
                
                self.ds3, self.ds3_w, self.ds3_b = deconv2d(tf.nn.relu(e3),
                                                        [self.batch_size,s4,s4, self.gf_dim * 2], name='g_ds3',
                                                        with_w=True)
                residual_connection_e2 = tf.add(self.ds3,e2)
                # print("e2 and residual_connection_e2:")
                # print(e2.shape)
                # print(residual_connection_e2.shape)
                
                self.ds2, self.ds2_w, self.ds2_b = deconv2d(tf.nn.relu(e2),
                                                        [self.batch_size,s2,s2, self.gf_dim], name='g_ds2',
                                                        with_w=True)
                residual_connection_e1 = tf.add(self.ds2,e1)
                # print("e1 and residual_connection_e1:")
                # print(e1.shape)
                # print(residual_connection_e1.shape)
                #-------------------残差连接-------------
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                        [self.batch_size, s128, s128, self.gf_dim * 8], name='g_d1',
                                                        with_w=True)
                d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, residual_connection_e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)
                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                        [self.batch_size, s64, s64, self.gf_dim * 8], name='g_d2',
                                                        with_w=True)
                
                d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2,residual_connection_e6], 3)
                #print(d2.shape)
                # d2 is (4 x 4 x self.gf_dim*8*2)
                #------------------------------
                
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                        [self.batch_size, s32, s32, self.gf_dim * 8], name='g_d3',
                                                        with_w=True)
                d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, residual_connection_e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)

                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                        [self.batch_size, s16, s16, self.gf_dim * 8], name='g_d4',
                                                        with_w=True)
                d4 = self.g_bn_d4(self.d4)
                d4 = tf.concat([d4, residual_connection_e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                        [self.batch_size, s8, s8, self.gf_dim * 4], name='g_d5',
                                                        with_w=True)
                d5 = self.g_bn_d5(self.d5)
                d5 = tf.concat([d5,residual_connection_e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)
                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                        [self.batch_size, s4, s4, self.gf_dim * 2], name='g_d6',
                                                        with_w=True)
                d6 = self.g_bn_d6(self.d6)
                d6 = tf.concat([d6, residual_connection_e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)
                self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                                                        [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
                d7 = self.g_bn_d7(self.d7)
                d7 = tf.concat([d7, residual_connection_e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                                                        [self.batch_size, s, s, self.output_c_dim], name='g_d8',
                                                        with_w=True)
                #print(self.d8.shape)
                # d8 is (256 x 256 x output_c_dim)
                return tf.nn.tanh(self.d8)
    
    '''              
    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                                                     [self.batch_size, s128, s128, self.gf_dim * 8], name='g_d1',
                                                     with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                                                     [self.batch_size, s64, s64, self.gf_dim * 8], name='g_d2',
                                                     with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                                                     [self.batch_size, s32, s32, self.gf_dim * 8], name='g_d3',
                                                     with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                                                     [self.batch_size, s16, s16, self.gf_dim * 8], name='g_d4',
                                                     with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                                                     [self.batch_size, s8, s8, self.gf_dim * 4], name='g_d5',
                                                     with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                                                     [self.batch_size, s4, s4, self.gf_dim * 2], name='g_d6',
                                                     with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                                                     [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                                                     [self.batch_size, s, s, self.output_c_dim], name='g_d8',
                                                     with_w=True)
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(self.d8)
    '''
    def save(self,epoch,checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s_%s_mutildiscrimitor_unet++" % (self.dataset_name, self.batch_size, self.output_size,epoch)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        #修改加载的模型
        #model_dir = "%s_%s_%s_%s_mutildiscrimitor——unet++" % (self.dataset_name, self.batch_size, self.output_size,epoch)
        model_dir = "%s_%s_%s_10_mutildiscrimitor_unet++" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print (checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Loading testing images ...")
        # sample_files = [glob('./datasets/{}/train/X/*-*-{}.jpg'.format(self.dataset_name, index+1)) for index in self.Image_usage]
        # sample_files = np.array(sorted(np.array(sample_files).flatten(), key=functools.cmp_to_key(compare)))
        #
        # sample_images = [process_X_hyper(sample_files[i*self.input_c_dim*self.batch_size:(i+1)*self.input_c_dim*self.batch_size], self.batch_size, is_grayscale=True)
        #           for i in range(int(self.data_nums/self.batch_size))]
        data_X = glob('./datasets/{}/test/X/*.jpg'.format(self.dataset_name))            #x_test file
        data_X = np.array(sorted(data_X, key=functools.cmp_to_key(compare)))
        data_Y = glob('./datasets/{}/test/Y/*.jpg'.format(self.dataset_name))             #y_test file
        data_Y = np.array(sorted(data_Y, key=functools.cmp_to_key(compare)))
        batch_idxs = min(len(data_Y), args.train_size) // self.batch_size       #when len(data_Y) =1 
        # print(len(data_X))
        # print(len(data_Y))
        print(batch_idxs)
        
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            
        
        for idx in range(batch_idxs):           #修改此部分代码
            #print(idx * self.batch_size * self.hyper_dims + 0 * self.hyper_dims + self.Image_usage)
            batch_files_X = np.array(
                [data_X[idx * self.batch_size * self.hyper_dims + eve_batch * self.hyper_dims + self.Image_usage]
                 for eve_batch in range(self.batch_size)]).flatten()     #self.batch_size
            batch_files_Y = data_Y[idx * self.batch_size: (idx + 1) *self.batch_size]  
            batch = process_hyper_data(batch_files_X, batch_files_Y,self.input_c_dim,self.image_size)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: batch}
            )
            save_images(samples, [self.batch_size, 1],                         #修改此处的保存文件名
                        './{}/{}/test_{:04d}.jpg'.format(args.test_dir,args.dataset_name,idx))
            # errD_fake = self.d_loss_fake.eval({self.real_data: batch})
            # errD_real = self.d_loss_real.eval({self.real_data: batch})
            # errG = self.g_loss.eval({self.real_data: batch})
            # epoch_dloss +=errD_fake+errD_real
            # epoch_gloss +=errG
        
        

    

