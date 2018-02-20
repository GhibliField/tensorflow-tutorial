"""
这个脚本使用placeholder和feed_dict来声明数据类型和喂数据
"""

import tensorflow as tf
input0=tf.placeholder(tf.float32)#placeholder的参数是（dtype,shape,name）
input1=tf.placeholder(tf.float32)

output=tf.multiply(input0,input1)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input0:3,input1:5}))