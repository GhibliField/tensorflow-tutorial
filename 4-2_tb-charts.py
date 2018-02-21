

"""
这个脚本是在4-1的基础上增加tensorboard的可视化功能,在tensorboard中的graph标签页显示图结构，在event和histogram增加折线图和直方图
"""
import tensorflow as tf
import numpy as np

def addlayer(inputs,in_size,out_size,n_layer,activation_func=None):
    layer_name='LAYER_%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('WEIGHTS'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/Weights',Weights)
        with tf.name_scope('BIASES'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+r'/biasess',biases)
        with tf.name_scope('WX_PLUS_B'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases

        if activation_func is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_func(Wx_plus_b,)#当你自己选择用 tensorflow 中的激励函数（activation function）的时候，tensorflow会默认添加名称。
        tf.summary.histogram(layer_name+'/output',outputs)
        return outputs

if __name__=='__main__':
    with tf.name_scope('INPUT'):#使用with tf.name_scope('inputs')可以将xs和ys包含进来，形成一个大的图层，
        # 图层的名字就是with tf.name_scope()方法里的参数。
        xs = tf.placeholder(tf.float32, [None, 1],name='X_INPUT')
        ys = tf.placeholder(tf.float32, [None, 1],name='Y_INPUT')
    # add hidden layer
    l1=addlayer(xs,1,10,1,activation_func=tf.nn.relu)
    # add output layer
    prediction=addlayer(l1,10,1,2,activation_func=None)
    with tf.name_scope('LOSS'):
        loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
        tf.summary.scalar('loss',loss)
    with tf.name_scope('TRAIN'):
        train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)#将上面‘绘画’出的图保存到一个目录中，以方便后期在浏览器中可以浏览。 这个方法中的第二个参数需要使用sess.graph ， 因此我们需要把这句话放在获取session的后面。 这里的graph是将前面定义的框架信息收集起来，然后放在logs/目录下面。
        merged=tf.summary.merge_all()#接下来， 开始合并打包。该方法会将我们所有的 summaries 合并到一起
        sess.run(init)
        for i in range(1000):
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
            if i%50==0:
                result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
                writer.add_summary(result,i)#i表示图表中采样的间隔
