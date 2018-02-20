import tensorflow as tf
import numpy as np
#创造一些数据点
x_data=np.random.rand(100).astype(np.float32)#生成100个（0,1）随机数组成numpy.ndarray数组
#float32是tf中大部分使用到的数据类型
y_data=x_data*0.1+0.3#自己设想的一个方程让模型来拟合。良好的拟合结果应该接近这个方程中的权重(weight)和偏置（bias）
#y_data的类型与x_data相同

#构建tf结构
Weights=tf.Variable(tf.random_uniform([1],-0.1,1.0))#在当前场景下，权重的shape是1个数，故使用随机初始化函数的时候定义为[1]，随机取值的空间在（-1,1）
biases=tf.Variable(tf.zeros([1]))#直接初始为0

y=Weights*x_data+biases

#目标函数
loss=tf.reduce_mean(tf.square(y-y_data))#计算预测与实际的差别,使用均方误差
#可想而知，一开始差别是很大的

#下面建立优化器来减少误差，这是每一次构建神经网络都需要做的事情
optimizer=tf.train.GradientDescentOptimizer(0.5)#最基础最原始的优化器
#优化器的参数是学习率，学习率是一个小于1大于0的数

train=optimizer.minimize(loss)

#以上虽然建立了变量，但是还没有在tf中初始化，使用下面的语句
init=tf.global_variables_initializer()
#构建tf结构结束

#创建会话
sess=tf.Session()
sess.run(init)#图必须在会话中执行,现在先在会话中初始化变量

for step in range(200):#令模型迭代200步
    sess.run(train)#在会话中训练
    if step%20==0:#每迭代20步打印一次权重和偏置
        print(step,sess.run(Weights),sess.run(biases))