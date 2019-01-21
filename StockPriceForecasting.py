import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,model_selection
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Importing dataset
data = pd.read_csv('msft_stockprices_dataset.csv', delimiter=',')
used_features = [ "High Price", "Low Price" , "Open Price" ,"Volume"]
X = np.array(data[used_features])
y = np.array(data["Close Price"])


# 将特征数据集分为训练集和测试集，除了最后 20% 个作为测试用例，其他都用于训练
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size = 0.2)

    
# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型
regr.fit(X_test, y_test)

# 用训练得出的模型进行预测
diabetes_y_pred = regr.predict(X_test)

#判断预测是否准确（ErrorTolerance=10%）
def judge(predict,aim):
    comRes =abs(predict-aim)/aim
    if comRes <0.1 :
        return True
    else:
        return False

#计算预测准确率
def getScore(diabetes_y_pred,y_test):
    size = len(y_test)
    rightCount = 0
    for i in range(0, size):  
        if judge(diabetes_y_pred[i],y_test[i]):
            rightCount=rightCount+1

    return rightCount/size

accuracy = regr.score(X_test,y_test)
print("accuracy=",accuracy*100,'%')

accuracy1 = getScore(diabetes_y_pred,y_test) ;
print("accuracy1=",accuracy1*100,'%')

# 将测试结果以图形的方式显示出来
xpos = []
for i in  range(len(y_test) ):
    xpos.append(i)
plt.plot(xpos, y_test, color='black', linewidth=3)
#plt.scatter(xi, y_test,  color='black')
plt.plot(xpos, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
