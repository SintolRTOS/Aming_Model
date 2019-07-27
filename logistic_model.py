# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:20:44 2019

@author: wangjingyi
"""

import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures # 导入多项式回归模型
from sklearn.externals import joblib
 
plt.figure()  # 实例化作图变量
plt.title('aming trans line points')  # 图像标题
plt.xlabel('aming point')  # x轴文本
plt.ylabel('aming value')  # y轴文本
plt.axis([30, 400, 100, 400])
plt.grid(True)  # 是否绘制网格线
 
# 训练数据（给定的阿明转化指数x和阿明转化后数据y）
X = [[49960],[44917],[37668],[34094],[32014],[30942],[29053],[27020],[26990],[25664],[23603],[23398],[23367],[23098],[22903],[22445],[21491],[21436]]
y = [[116104],[96288],[70703],[59376],[53187],[50114],[44893],[39557],[39479],[36163],[31259],[30788],[30716],[30105],[29665],[28640],[26558],[26440]]
 
# 做最终效果预测的样本
X_test = X  # 用来做最终效果测试
y_test = y  # 用来做最终效果测试

# 绘制散点图
plt.scatter(X, y, marker='*',color='blue',label='阿明指数转化样本')

# 线性回归
model = LinearRegression()
model.fit(X,y)
# 模型拟合效果得分
print('一元线性回归 r-squared',model.score(X_test,y_test))
x2=[[21436],[49960]] # 所绘制直线的横坐标x的起点和终点
y2=model.predict(x2)
plt.plot(x2,y2,'g-')  # 绿色的直线

# 二次多项式回归
# 实例化一个二次多项式特征实例
quadratic_featurizer=PolynomialFeatures(degree=2)
# 用二次多项式对样本X值做变换
X_train_quadratic = quadratic_featurizer.fit_transform(X)
# 创建一个线性回归实例
regressor_model=linear_model.LinearRegression()
# 以多项式变换后的x值为输入，带入线性回归模型做训练
regressor_model.fit(X_train_quadratic,y)
# 设计x轴一系列点作为画图的x点集
xx=np.linspace(21436,49960,1000)
# 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_predict = regressor_model.predict(xx_quadratic)
# 用训练好的模型作图
plt.plot(xx, yy_predict, 'r-')
X_test_quadratic = quadratic_featurizer.transform(X_test)
print('二次回归  r-squared', regressor_model.score(X_test_quadratic, y_test))

#三次多项式回归
cubic_featurizer = PolynomialFeatures(degree=3)
X_train_cubic = cubic_featurizer.fit_transform(X)
regressor_cubic = LinearRegression()
regressor_cubic.fit(X_train_cubic, y)
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_cubic.predict(xx_cubic))
X_test_cubic = cubic_featurizer.transform(X_test)
print('三次回归     r-squared', regressor_cubic.score(X_test_cubic, y_test))
plt.show()  # 展示图像

#保存模型
joblib.dump(regressor_cubic, './model/modelFile.pkl')

#加载模型(--加载)
clf3 = joblib.load('./model/modelFile.pkl')
#使用模型预测检测准确度
print('使用模型预测准确度:',regressor_cubic.score(X_test_cubic,y_test))
#使用模型预测计算
##(--加载)
#cubic_featurizer = PolynomialFeatures(degree=3)
##(--加载)
#X_train_cubic2 = cubic_featurizer.fit_transform(X)
##(--加载)
#X_test_cubic2 = cubic_featurizer.transform(X_test)
##(--加载)
#print('使用模型预测计算值:',clf3.predict(X_test_cubic2))