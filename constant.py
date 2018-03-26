# tf.constant(
#     value,
#     dtype=None,
#     shape=None,
#     name='Const',
#     verify_shape=False
# )
# 功能说明：根据 value 的值生成一个 shape 维度的常量张量
#
# 参数列表：
# value	,常量数值或者 list	输出张量的值
# dtype	,dtype	输出张量元素类型
# shape,1 维整形张量或 array	输出张量的维度
# name	,	string	张量名称
# verify_shape,	Boolean	检测 shape 是否和 value 的 shape 一致，若为 Fasle，不一致时，会用最后一个元素将 shape 补全
#

#!/usr/bin/python

import tensorflow as tf
import numpy as np
a = tf.constant([1,2,3,4,5,6],shape=[2,3])
b = tf.constant(-1,shape=[3,2])
c = tf.matmul(a,b)

e = tf.constant(np.arange(1,13,dtype=np.int32),shape=[2,2,3])
f = tf.constant(np.arange(13,25,dtype=np.int32),shape=[2,3,2])
g = tf.matmul(e,f)
with tf.Session() as sess:
    print (sess.run(a))
    print ("##################################")
    print (sess.run(b))
    print ("##################################")
    print (sess.run(c))
    print ("##################################")
    print (sess.run(e))
    print ("##################################")
    print (sess.run(f))
    print ("##################################")
    print (sess.run(g))
