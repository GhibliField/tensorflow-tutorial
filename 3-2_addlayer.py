"""
这个脚本通过迭代降低损失函数，最终构建一个神经网络结构：一个输入层，一个隐含层，一个输出层
"""
import tensorflow as tf
import numpy as np

def addlayer(inputs,in_size,out_size,activation_func=None):#in_size表示输入矩阵的列数
    # inputs: (n_samples, n_features) 特征数就是该层连接的输入的神经元的个数
    # weights: (n_features, neurons) neurons就是该层连接的输出神经元个数
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))#产因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    Wx_plus_b=tf.matmul(inputs,Weights)+biases#用 `+` 和 tf.add() 是一样的作用
    #注意：矩阵论中接触较多的是Wx形式，而tf的源码的形式是xW.
    #Weights.shape=(n_features, n_outputs), inputs.shape = (n_samples, n_features)
    #所以, i*W = (n_samples, n_outputs),
    print(Wx_plus_b.shape)
    if activation_func is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_func(Wx_plus_b)
    return outputs

if __name__=='__main__':
    #构建图结构
    # 利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1。
    xs = tf.placeholder(tf.float32, [None, 1])#一个列向量
    ys = tf.placeholder(tf.float32, [None, 1])
    # add hidden layer
    l1=addlayer(xs,1,10,activation_func=tf.nn.relu)#第一层，输入层到隐含层，输入用一个神经元表示，隐层有10个神经元
    # add output layer
    prediction=addlayer(l1,10,1,activation_func=None)#第二层，隐层到输出层，输出一个神经元表示
    #这里的输入层只有一个属性， 所以我们就只有一个输入；隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元；
    # 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。 所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。
    #计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    #reduce_sum中的reduction_indices=[1]表示按行求和，reduction_indices=[0]表示按列求和
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 下面创造一些数据，利用添加噪声让数据看起来更真实
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]#linspace用于创造[-1,1]内间隔相等的300个数组成的数列
    #[:, np.newaxis]将一个以为数列转换为二维的矩阵，每一行为元数列中的一个元素
    noise = np.random.normal(0, 0.05, x_data.shape)  # 制造符合正态分布的噪声，0表示正态分布的均值，0.05表示方差
    y_data = np.square(x_data) - 0.5 + noise

    #下面初始化变量并将我们构建的图执行起来
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
            if i%50==0:
                print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
