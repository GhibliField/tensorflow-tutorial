import tensorflow as tf
"""
tf.nn.conv2d函数
    input,tensor	是一个 4 维的 tensor，即 [ batch, in_height, in_width, in_channels ]（若 input 是图像，[ 训练时一个 batch 的图片数量, 图片高度, 图片宽度, 图像通道数 ]）
    filter,tensor	是一个 4 维的 tensor，即 [ filter_height, filter_width, in_channels, out_channels ]（若 input 是图像，[ 卷积核的高度，卷积核的宽度，图像通道数，卷积核个数 ]）,filter 的 in_channels 必须和 input 的 in_channels 相等
    strides,列表	长度为 4 的 list，卷积时候在 input 上每一维的步长，一般 strides[0] = strides[3] = 1
    padding,string	只能为 " VALID "，" SAME " 中之一，这个值决定了不同的卷积方式。VALID 丢弃方式；SAME：补全方式
    use_cudnn_on_gpu,bool	是否使用 cudnn 加速，默认为 true
    data_format,string	只能是 " NHWC ", " NCHW "，默认 " NHWC "
    name,string	运算名称
"""
a=tf.constant([1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0],dtype=tf.float32,shape=[1,5,5,1])
b = tf.constant([1,0,1,0,1,0,1,0,1],dtype=tf.float32,shape=[3,3,1,1])
c = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='VALID')
d = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    print ("c shape:")
    print (c.shape)
    print ("c value:")
    print (sess.run(c))
    print ("d shape:")
    print (d.shape)
    print ("d value:")
print (sess.run(d))