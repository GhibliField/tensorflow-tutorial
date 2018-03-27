"""
这个脚本定义了神经网络的训练过程
"""

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

#配置神经网络参数
BATCH_SIZE=100#一个训练batch中的训练数据个数，数字越小，训练过程越接近原生的SGD;数字越大，训练越接近梯度下降
LEARNING_RATE_BASE=0.8#基础的学习率，即初始值
LEARNING_RATE_DECAY=0.99#学习率的衰减率
REGULARIZATION_LAMBDA=0.0001#描述模型复杂度的正则化项在损失函数中的稀疏
TRAINING_STEPS=30000#训练轮数
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
#模型保存的路径和用户名
MODEL_SAVE_PATH="pickled_variables"
MODEL_NAME="model.ckpt"
def train(mnist):
    #定义输入输出placeholder
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    y=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-input")
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_LAMBDA)#定义正则化函数
    #直接使用mnist_inference中定义的前向传播过程
    pred=mnist_inference.inference(x,regularizer)
    #定义保存训练轮数的变量。这个变量不需要使用滑动平均值，所以这里指定这个变量为不可训练的变量。在使用tf训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    globel_step=tf.Variable(0,trainable=False)

    #定义损失函数、学习率、滑动平均操作及训练过程
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,globel_step)#初始化滑动平均类，加快训练早期变量的更新速度
    #在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op=variable_averages.apply(tf.trainable_variables())#tf.trainable_variables返回的就是集合GraphKeys.TRAINABLE_VARIABELS中的元素。这个集合的元素就是所有可训练的参数。
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=tf.argmax(y,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)#计算在当前batch中所有样例的交叉熵平均值
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))#总损失等于交叉熵损失和正则化损失的和
    #下面设定指数衰减的学习率
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,#基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
                                             globel_step,
                                             mnist.train.num_examples/BATCH_SIZE,#过完所有的训练数据需要的迭代次数
                                             LEARNING_RATE_DECAY)#学习率的衰减速度
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=globel_step)
    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数又要更新每一个参数的滑动平均值。为了完成多个操作，TF提供了
    #tf.control.dependencies和tf.group两种机制。下面的程序等价于
    #train_op=tf.group(train_step,variables_averages_op)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name='train')
    #初始化TensorFlow持久化类
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,globel_step],feed_dict={x:xs,y:ys})
            #每1000轮保存一次模型
            if i % 1000==0:
                #输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
                #在验证集上的正确率信息会有一个单独的程序来生成
                print("After %d training step(s), loss on training batch is %g."%(step,loss_value))
                #保存当前的模型。注意这里给出了global_step参数，这样让每一个被保存的模型的文件名末尾加上训练的轮数，比如“model.cplt-1000”表示训练1000轮之后得到的模型
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=globel_step)


def main(argv=None):
    mnist=input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)

if __name__=='__main__':
    tf.app.run()