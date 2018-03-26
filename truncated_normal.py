#  tf..truncated_normal(
#     shape,
#     mean=0.0,
#     stddev=1.0,
#     dtype=tf.float32,
#     seed=None,
#     name=None
# )
# 功能说明：产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]。
#
# 参数列表：
# shape	,1维整形张量或 array	输出张量的维度
# mean	,1维张量或数值	均值
# stddev,1维张量或数值	标准差
# dtype	,dtype	输出类型
# seed	,数值	随机种子，若 seed 赋值，每次产生相同随机数
# name	,string	运算名称

import tensorflow as tf
initial = tf.truncated_normal(shape=[3,3], mean=0, stddev=1)
print(tf.Session().run(initial))