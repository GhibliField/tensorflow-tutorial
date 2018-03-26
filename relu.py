
"""
 tf.nn.relu(
    features,
    name=None
)
参数列表：

features,tensor	是以下类型float32, float64, int32, int64, uint8, int16, int8, uint16, half
name,string	运算名称
"""
import tensorflow as tf

a = tf.constant([1,-2,0,4,-5,6])
b = tf.nn.relu(a)
with tf.Session() as sess:
    print (sess.run(b))