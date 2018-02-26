"""
这个脚本定义了前向传播的过程以及神经网络中的参数。无论是训练时还是测试时，都可以直接调用inference这个函数，而不用关心具体的神经网络结构

"""
import tensorflow as tf

#定义神经网络结构相关参数
INPUT_NODE=784#输入层的节点数，对于MNIST数据集，这个就是图片的像素
OUTPUT_NODE=10#输出层的节点数。这个等于类别的数目
LAYER1_NODE=500#隐藏层的节点数。这里使用只有一个隐含层的网络结构作为样例

#下面通过tf.get_variable函数来获得变量。在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值。
#而且更方便的是，因为可以在变量加载时将滑动平均变量重命名，所以可以直接通过相同的名字在训练时使用变量本身，而在测试时使用变量的滑动平均值。
#在这个函数中也会将变量的正则化损失加入损失集合。
def get_weight_variable(shape,regularizer):
    weights=tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="weights")
    #当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合。
    #在这里使用add_to_collection函数将一个张量加入一个集合，而这个集合的名称为losses。这是自定义的集合，不在TF自动管理的集合列表中
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

#定义神经网络的前向传播过程
#inference函数给定神经网络的输入和所有的参数，计算神经网络的前向传播的结果。在这里定义了一个使用ReLu激活函数的三层全连接层神经网络。
#通过加入隐藏层实现多层网络结构。通过ReLu实现去线性化。
def inference(input_tensor,regularizer):
    #声明隐藏层的参数并完成前向传播过程
    with tf.variable_scope("layer1"):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.Variable(tf.zeros([LAYER1_NODE],name="biases"))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    #类似的声明输出层的参数并完成前向传播过程
    with tf.variable_scope("layer2"):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.Variable(tf.zeros([OUTPUT_NODE], name="biases"))
        layer2=tf.matmul(layer1,weights)+biases
    #返回最后的前向传播的节骨
    return layer2