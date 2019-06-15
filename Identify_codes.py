#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from PIL import Image

input_dir = './splited_imgs'
out_path = './identified_imgs'
model_dir = './train_model'
if not os.path.exists(out_path):
    os.makedirs(out_path)

w = 14
h = 27

chars = []

for (path, dirnames, filenames) in os.walk(input_dir):
    for dirname in dirnames:
        chars.append(dirname)

x = tf.placeholder(tf.float32, [None, h, w, 1])
y_ = tf.placeholder(tf.float32, [None, len(chars)])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,1,32]) # 卷积核大小(3,3)， 输入通道(1)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([2*8*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 2*8*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, len(chars)])
    bout = weightVariable([len(chars)])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnnLayer()
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint(model_dir + '/.')) 

def identify_char(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    w, h, z = image.shape
    # 灰度
    GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(GrayImage,w,h, cv2.THRESH_TOZERO) 
    imgs = []
    imgs.append(thresh)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0], 27, 14, 1)
    imgs = imgs.astype('float32') / 255.0
    res = sess.run(predict, feed_dict={x: imgs, keep_prob_5:1.0, keep_prob_75: 1.0})  
    return chars[res[0]]

def identify_image(img):
    result = []
    w, h = img.size

    box = (3, 0, 17, h)
    result.append(identify_char(img.crop(box)))

    box = (15, 0, 29, h)
    result.append(identify_char(img.crop(box)))

    box = (27, 0, 41, h)
    result.append(identify_char(img.crop(box)))

    box = (39, 0, 53, h)
    result.append(identify_char(img.crop(box)))

    return result

if __name__ == '__main__':
    img = Image.open('./code_imgs/b8uj.jpg')
    result = identify_image(img)

    print(result)



