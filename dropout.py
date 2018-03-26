# tf.nn.dropout(
#     x,
#     keep_prob,
#     noise_shape=None,
#     seed=None,
#     name=None
# )
#
# 参数列表：
#
# x,tensor	输出元素是 x 中的元素以 keep_prob 概率除以 keep_prob，否则为 0
# keep_prob,scalar Tensor	dropout 的概率，一般是占位符
# noise_shape,tensor	默认情况下，每个元素是否 dropout 是相互独立。如果指定 noise_shape，若 noise_shape[i] == shape(x)[i]，该维度的元素是否 dropout 是相互独立，若 noise_shape[i] != shape(x)[i] 该维度元素是否 dropout 不相互独立，要么一起 dropout 要么一起保留
# seed,数值	如果指定该值，每次 dropout 结果相同
# name,string	运算名称

import tensorflow as tf

a = tf.constant([1,2,3,4,5,6],shape=[2,3],dtype=tf.float32)
b = tf.placeholder(tf.float32)
c = tf.nn.dropout(a,b,[2,1],1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(c,feed_dict={b:0.75}))