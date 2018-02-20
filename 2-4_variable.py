"""
这个脚本是通过一个循环让一个变量依次加1
"""
import tensorflow as tf

state=tf.Variable(0,name='counter')#在图中的变量必须声明他是Varialble才算完成声明变量
#变量可以给定一个初始值，可以起别名
#print(state.name) 将输出：counter:0  注意，这里的0不是初始值0，而是表示这个变量是声明的第0个变量
one=tf.constant(1)

new_value=tf.add(state,one)#tf中的加操作
update=tf.assign(state,new_value)#tf中的赋值操作，将new_value的值赋值给state

#但凡在图结构中声明了变量，在创建会话前必须进行初始化和激活
init=tf.global_variables_initializer()#全局变量初始化器
with tf.Session() as sess:
    sess.run(init)#激活所有变量
    for i in range(3):
        sess.run(update)#执行图中得到update操作
        print(sess.run(state))#如果使用print(state)是不会有用的