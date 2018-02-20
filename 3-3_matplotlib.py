"""
这个脚本通过迭代降低损失函数，最终构建一个神经网络结构：一个输入层，一个隐含层，一个输出层
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def addlayer(inputs,in_size,out_size,activation_func=None):#in_size表示输入矩阵的列数
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))#产生符合正态分布的随机数
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#推荐不要初始化为0，所以这边加上0.1
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_func is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_func(Wx_plus_b)
    return outputs

if __name__=='__main__':
    #构建图结构
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1])#一个列向量
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer
    l1=addlayer(xs,1,10,activation_func=tf.nn.relu)#第一层，输入层到隐含层，输入用一个神经元表示，隐层有10个神经元
    # add output layer
    prediction=addlayer(l1,10,1,activation_func=None)#第二层，隐层到输出层，输出一个神经元表示
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    #reduce_sum中的reduction_indices=[1]表示按行求和，reduction_indices=[0]表示按列求和
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 下面创造一些数据，利用添加噪声让数据看起来更真实
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]#linspace用于创造[-1,1]内间隔相等的300个数组成的数列
    #[:, np.newaxis]将一个以为数列转换为二维的矩阵，每一行为元数列中的一个元素
    noise = np.random.normal(0, 0.05, x_data.shape)  # 制造符合正态分布的噪声，0表示正态分布的均值，0.05表示方差
    y_data = np.square(x_data) - 0.5 + noise

    # 构建图形，用散点图描述真实数据之间的关系。 （注意：plt.ion()用于连续显示。）
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    #下面初始化变量并将我们构建的图执行起来
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            # training
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:#每隔50次训练刷新一次图形
                # to visualize the result and improvement
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                # 用红色、宽度为5的线来显示我们的预测数据和输入之间的关系，并暂停0.1s。
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                plt.pause(0.1)


