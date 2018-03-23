#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:37:09 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import os
data_folder = os.path.join(os.path.expanduser('~'), 'Desktop/Master File/practice of python/Data', 'cifar-10')
batch1_filename = os.path.join(data_folder, 'data_batch_1')

#打开 pickle 格式的图像数据
import pickle
def unpickle(filename):
    with open (filename, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

batch1 = unpickle(batch1_filename)

#示例
image_index = 100
image = batch1['data'][image_index]

image = image.reshape((32, 32, 3), order='F') #32像素 * 32像素 * 3，红绿蓝三原色

import numpy as np

image = np.rot90(image, -1)  #矩阵旋转，-1 是顺时针90

from matplotlib import pyplot as plt
plt.imshow(image)



#####Theano
import theano
from theano import tensor as T

a = T.dscalar()
b = T.dscalar()
c = T.sqrt(a **2 + b ** 2)

f = theano.function([a, b], c)
f(3, 4)


#####Deep-Learning try
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.astype(np.float32)
y_true = iris.target.astype(np.int32)   #lasagne 需求 32 类型

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, random_state=14)

import lasagne

input_layer = lasagne.layers.InputLayer(shape=(10, X.shape[1])) #每一批输入10个

hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=12, nonlinearity=lasagne.nonlinearities.sigmoid)
#共12个神经元， 使用sigmod函数

output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)
#3分类，3个神经元，softmax函数常作输出

#Theano声明变量
net_input = T.matrix('net_input')  #输入
net_output = lasagne.layers.get_output(output_layer, net_input)   #输出
true_output = T.ivector('true_output')   #真实输出

loss = T.mean(T.nnet.categorical_crossentropy(net_output, true_output))
#损失函数， 类别交叉熵

#调参
all_params = lasagne.layers.get_all_params(output_layer)
updates = lasagne.updates.sgd(loss, all_params, learning_rate=0.1)

#训练
import theano
train = theano.function([net_input, true_output], loss, updates=updates)   #train
get_output = theano.function([net_input], net_output)   #获取输出

for n in range(1000):   #训练1000次
    train(X_train, y_train)
    
y_output = get_output(X_test)  #预测结果

#观察效果F值
import numpy as np
y_pred = np.argmax(y_output, axis=1)

from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='micro'))

#封装 lasagne  --- nolearn
#验证码的例子，调整了数字格式为32，对 X 归一化
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize
from skimage import transform as tf
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

def create_captcha(text, shear=0, size=(100, 30)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"Coval-Black.otf", 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()


def segment_image(image):
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x,start_y:end_y])
    if len(subimages) == 0:
        return [image,]
    return subimages

random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear=shear, size=(30, 30)), letters.index(letter)
dataset, targets = zip(*(generate_sample(random_state) for i in range(3000)))
dataset = np.array(dataset, dtype='float')
targets =  np.array(targets)

onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0],1))
y = y.todense().astype(np.float32)

dataset = np.array([resize(segment_image(sample)[0], (30, 30)) for sample in dataset])
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
X = X / X.max()
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=0.9, random_state=14)


##用 nolearn 重建
from lasagne import layers
#基本结构
layers = [
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ]

from lasagne import updates
from nolearn.lasagne import NeuralNet      
from lasagne.nonlinearities import sigmoid, softmax
#构建
net1 = NeuralNet(layers=layers,
                 input_shape=X.shape,
                 hidden_num_units=100,
                 output_num_units=26,
                 hidden_nonlinearity=sigmoid,
                 output_nonlinearity=softmax,
                 hidden_b=np.zeros((100,), dtype=np.float64),
                 #偏置神经元，它们位于隐含层，一直处于激活状态。偏置神经元对于网络的训 练很重要，
                 #神经元激活后，可以对问题做更有针对性的训练，以消除训练中的偏差。
                 #举个简单的例子，如果预测结果总是偏差4，我们可以使用4偏置值抵消偏差。
                 #我们设置的偏置神经元就能 起到这样的作用，训练得到的权重决定了偏置值的大小。
                 update=updates.momentum,
                 update_learning_rate=0.9,
                 update_momentum=0.1,  #冲量
                 regression=True,  #虽然是分类，但回归在这的效果好
                 max_epochs=1000)

#训练
net1.fit(X_train, y_train)
#test1
y_pred = net1.predict(X_test)
y_pred = y_pred.argmax(axis=1)
assert len(y_pred) == len(X_test)
if len(y_test.shape) > 1:
    y_test = y_test.argmax(axis=1)
    
print(f1_score(y_test, y_pred, average='micro'))



##CIFAR

#构建数据集
import os
import numpy as np
batches = []
for i in range(1, 6):
    batch_filename = os.path.join(data_folder, 'data_batch_{}'.format(i))  #导入 1-6
    batches.append(unpickle(batch_filename))
    

X = np.vstack([batch['data'] for batch in batches])
X = np.array(X) / X.max()
X = X.astype(np.float32)

from sklearn.preprocessing import OneHotEncoder
y = np.hstack(batch['labels'] for batch in batches).flatten()
y = OneHotEncoder().fit_transform(y.reshape(y.shape[0], 1)).todense()
y = y.astype(np.float32)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)    #reshape(-1,...) 计算完后面维度后自动生成 -1

#bulid
from lasagne import layers
layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ]

from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import sigmoid, softmax
nnet = NeuralNet(layers=layers,
                 input_shape=(None, 3, 32, 32),  #None表示nolearn每次使用默认数量的图像数据进行训练                
                 conv1_num_filters=32,
                 conv1_filter_size=(3, 3),
                 conv2_num_filters=64,
                 conv2_filter_size=(2, 2),
                 conv3_num_filters=128,
                 conv3_filter_size=(2, 2),    #我也不知道参数咋调
                 pool1_pool_size=(2,2),
                 pool2_pool_size=(2,2),
                 pool3_pool_size=(2,2),
                 hidden4_num_units=500,
                 hidden5_num_units=500,
                 output_num_units=10,                 
                 output_nonlinearity=softmax,
                 update_learning_rate=0.01,   #update是会更新的学习速率
                 update_momentum=0.9,
                 regression=True,
                 max_epochs=100,
                 verbose=1)  #每步训练都会输出结果，还能输出每一步训练所花的时间

#训练
nnet.fit(X_train, y_train)

#测试
from sklearn.metrics import f1_score
y_pred = nnet.predict(X_test)
print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro'))


# 3 步 f1 0.1419    100步 f1 0.599 提升还是很明显的























