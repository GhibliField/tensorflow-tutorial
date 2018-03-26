# tf.placeholder(
#     dtype,
#     shape=None,
#     name=None
# )
# 功能说明：是一种占位符，在执行时候需要为其提供数据
#
# 参数列表：
# dtype	,	dtype	占位符数据类型
# shape	,	1 维整形张量或 array	占位符维度
# name	,	string	占位符名称

#!/usr/bin/python

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[None,3])
y = tf.matmul(x,x)
with tf.Session() as sess:
    rand_array = np.random.rand(3,3)
    print(sess.run(y,feed_dict={x:rand_array}))