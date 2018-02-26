"""
这给脚本做一个没有隐层的全连接的神经网络实现MNIST任务
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#一共有10个类，使用十比特的one-hot来表示类别


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pred=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.argmax(v_ys,1))#预测结果（softmax的结果）中最大值所在索引等于真实验证数据真实分类的one-hot最大值（也就是一个1）的情况
    #tf.argmax()将返回其第一个参数在某一维度上的最大值所在索引号，0表示按列，1表示按行
    #tf.equal()返回一个shape与其第一个参数相同的矩阵或向量，其每隔元素的值是两个参数对应位置上数值是否相同的判断结果
    #tf.cast()用于将第一个参数转换成为第二个参数指定的数据类型，返回转换后的数据
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs: v_xs, ys: v_ys})
    return result

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

#定义一层神经网络层
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
#原始的神经网络的输出不是一个概率分布，要想使用交叉熵，就需要将这个原始输出转换为概率分布的形式，使用softmax来将原始输出转换为概率分布
#定义损失函数为交叉熵
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_steps=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer=tf.summary.FileWriter('logs/',sess.graph)
    merged=tf.summary.merge_all()
    STEPS=1000#设置训练轮数
    for i in range(STEPS):
        #为了方便使用随机梯度下降，使用mnist.train.next_batch()从所有的数据中读取一小部分作为一个训练batch
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次从数据集载入一小批数据
        sess.run(train_steps,feed_dict={xs:batch_xs,ys:batch_ys})
        if i%50==0:#每隔50轮计算准确率
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
