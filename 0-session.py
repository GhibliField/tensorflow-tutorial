"""
这个脚本执行的是两个常量矩阵相乘

"""

import tensorflow as tf

matrix0=tf.constant([[3,3]])#tf中定义常量的方法constant
matrix1=tf.constant([[2],[2]])

product=tf.matmul(matrix0,matrix1)#这里定义的是一个点乘操作，在构建图的过程中尚不没有声明操作的结果赋值给谁.numpy中时np.dot(m1,m2)
#以上，图构建完毕

#下面将图放入到会话中执行
#方法一：
# sess=tf.Session()
# result=sess.run(product)#操作的结果赋值给result
# print(result)
# sess.close()

#方法二
with tf.Session() as sess:
    result=sess.run(product)
    print(result)