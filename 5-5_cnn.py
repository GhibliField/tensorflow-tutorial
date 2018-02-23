
"""
这个脚本使用两层卷积来完成MNIST这个分类任务
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    #tf.truncated_normal用于产生服从正太分布的随机数，stddev用于设置标准差，均值默认为0
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    #tf.nn.conv2d参数
    # conv2d(
    #     input,4D张量，维度分别为[batch大小, 输入height, 输入width, 输入通道数（厚度/depth）]
    #     filter,4D张量，这个张量的维度分别为[filter （ kernel ）的height, filter 的width, 输入通道数（厚度/depth）, 输出通道数]
    #     strides,长度为4的1D张量，即一个整数数列，定义filter在input上每一个维度上的步进数
    #     padding,有"SAME"和"VALID"两种可选，选择SAME则在input边界时通过补零，是的feature map的大小与input相同
    #     use_cudnn_on_gpu=True,
    #     data_format='NHWC', 数据格式，定义各个维度的表示顺序，NHWC表示数据格式顺序为: [batch, height, width, channels]
    #     dilations=[1, 1, 1, 1],长度为4的1D张量，设置input每个维度的膨胀系数
    #     name=None
    # )
    # strides[0] 和 strides[3] 必须保持等于1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #tf.nn.max_pool参数
    # max_pool(
    #     value,4D张量，
    #     ksize,长度为4的1D张量，即一个整数数列，定义input各个维度上进行滑动的窗口大小
    #     strides,长度为4的1D张量，即一个整数数列，定义input各个维度的滑动窗口的滑动步进数
    #     padding,有"SAME"和"VALID"两种可选
    #     data_format='NHWC',
    #     name=None
    # )
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#tf.reshape(tensor,shape)用于将第一个参数tensor转换成第二个参数shape设置的形状，哪一个维度为-1，那么那个维度实际的大小==(tensor总的元素个数)/(shape中非-1元素的乘积)
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # filter的height*width== 5x5, 输入通道数为1, 设置输出通道数为32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # filter的height*width==5x5, 输入通道数为32, 设置输出通道数为64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7x7x64

## 定义全连接层fc1  ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#将张量扁平化
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#dropout

## 定义最后一层输出层fc2  ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#多分类时使用softmax


# 定义交叉熵为损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
learning_rate=1e-4
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy(
                mnist.test.images[:1000], mnist.test.labels[:1000]))

