"""
这个脚本使用基本的LSTM完成MNIST任务。这个 RNN 总共有 3 个组成部分 ( input_layer, cell, output_layer)

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
tf.set_random_seed(1)
#超参数
lr=0.001
training_iters=100000
batch_size=128
n_inputs=28#每行有28个像素
n_steps=28#每张图片有28行
n_hidden_units=128#两个隐层的单元数，自定义
n_classes=10 #类别数

#数据的placeholder
X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
Y=tf.placeholder(tf.float32,[None,n_classes])

#定义权重和偏差
weights={
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases={
    'in':tf.Variable(tf.random_normal([n_hidden_units,])),
    'out':tf.Variable(tf.random_normal([n_classes,]))
}

#定义网络结构
def RNN(X,weights,biases):
    #cell之前的隐层
    X=tf.reshape(X,[-1,n_inputs])
    X_in=tf.matmul(X,weights['in'])+biases['in']
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    #rnn
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)#这个tuple包含了c_state和mstate
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
    #cell之后的隐层
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
    return results

pred=RNN(X,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    step=0
    while step*batch_size<training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_op,feed_dict={X:batch_xs,Y:batch_ys})
        if step%20==0:
            print(sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys}))
        step+=1

