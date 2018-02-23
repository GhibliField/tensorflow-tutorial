"""
这个脚本使用placeholder和feed_dict来声明数据类型和喂数据
TenorFlow提供了placeholder机制用于提供输入数据。
placeholder相当于定义了一个位置，这个位置中的数据在程序运行时再指定。
这样在程序中就不需要生成大量常\变量来提供输入数据，而只需要将数据通过placeholder传入计算图。
"""

import tensorflow as tf
input0=tf.placeholder(tf.float32)#placeholder的参数是（dtype,shape,name）
input1=tf.placeholder(tf.float32)

output=tf.multiply(input0,input1)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input0:3,input1:5}))