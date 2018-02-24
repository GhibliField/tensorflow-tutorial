"""
这个脚本用于将定义好的变量保存到本地文件当中，然后在从本地文件中读取先前保存的变量
"""
import tensorflow as tf
import numpy as np

#定义要保存的变量
W=tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32,name='Weights')
b=tf.Variable([[0.1,0.2,0.3]],dtype=tf.float32,name='biases')

saver=tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    save_path=saver.save(sess,'pickled_variables/save_00.ckpt')#返回保存路径
    print("variables save to path:",save_path)

#下面从文件中读取变量
#读取之前要定义好同名变量的shape和数据类型
# W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='Weights')
# b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
#
# saver=tf.train.Saver()
# #注意：提取已经保存在文件中的变量时不再需要对变量进行初始化
# with tf.Session() as sess:
#     saver.restore(sess,"pickled_variables/save_00.ckpt")
#     print("Weights:",sess.run(W))
#     print("biases",sess.run(b))