#!/usr/bin/python
# -*- coding: utf-8 -*
"""
训练模型并和 sklearn 库线性回归模型对比
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from linear_regression_model import linearRegressionModel as lrm

if __name__ == '__main__':
    x, y = make_regression(7000)
    x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.5)
    print(x_train.shape)#(3500, 100)
    print(y_train.shape)#(3500,)
    y_lrm_train = y_train.reshape(-1, 1)
    y_lrm_test = y_test.reshape(-1, 1)
    print(y_lrm_train.shape)#(3500, 1)
    linear = lrm(x.shape[1])#参数是输入的特征数目
    linear.train(x_train, y_lrm_train,x_test,y_lrm_test)
    y_predict = linear.predict(x_test)
    print("Tensorflow R2: ", r2_score(y_predict.ravel(), y_lrm_test.ravel()))
    print(y_lrm_test.ravel().shape)#(3500,)

    lr = LinearRegression()
    y_predict = lr.fit(x_train, y_train).predict(x_test)
    print("Sklearn R2: ", r2_score(y_predict, y_test)) #采用r2_score评分函数