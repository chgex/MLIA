'''
Author: liubai
Date: 2021-02-03
LastEditTime: 2021-02-03
'''

import numpy as np
import time

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
    


def gradAscent(traindataArr,trainlabelArr):
    # 梯度上升，
    # 一次迭代遍历所有数据文件
    """
    参数：训练集，标签集
    return： weights
    """
    # w_0*x_0+w_1*x_1+w_2*x_2+...
    # 填充x1=1,则w_0等价于截距b
    for i in range(len(traindataArr)):
        traindataArr[i].append(1)
    # 转为数组类型，方便计算
    traindataArr=np.array(traindataArr)
    trainlabelArr=np.array(trainlabelArr)
    # shape为(100,)与shape为（100,1)区别：
    # 前者在矩阵运算时候可能发生广播，所以要reshape一下
    trainlabelArr=trainlabelArr.reshape(trainlabelArr.shape[0],1)
    
    # 初始化参数w，维度为(样本维数,1),即列向量
    # w=np.zeros(traindataArr.shape[1]),维度为：(1,样本维数)
    w=np.ones( (traindataArr.shape[1],1) )
    
    # 步长
    alpha=0.001
    # 迭代次数
    iter=500
    # iter次随机梯度上升
    for i in range(iter):
        # print("iter:%d /%d"%(i,iter))
        # 每一次迭代，都遍历所有样本
        h=sigmoid(np.dot(traindataArr,w))
        error=trainlabelArr-h
        w=w+ alpha* np.dot(traindataArr.T,error)
    # 返回weights 
    return w

def randGradAscent(traindataArr,trainlabelArr):
    # 随机梯度上升，
    # 每遍历一个数据文件就更新一次参数
    """
    参数：训练集，标签集
    return： weights
    """
    # w_0*x_0+w_1*x_1+w_2*x_2+...
    # 填充x1=1,则w_0等价于截距b
    for i in range(len(traindataArr)):
        traindataArr[i].append(1)
    # 转为数组类型，方便计算
    traindataArr=np.array(traindataArr)
    trainlabelArr=np.array(trainlabelArr)
    # shape为(100,),  取值trainlabelArr[1]=1.0
    # shape为(100,1), 取值trainlabelArr[1]=[1.0]
    # 注意类型
    trainlabelArr=trainlabelArr.reshape(trainlabelArr.shape[0],1)
    
    # 初始化参数w，维度为(样本维数,1),即列向量
    w=np.ones( (traindataArr.shape[1],1) )
    # 步长
    alpha=0.001
    # 迭代次数
    iter=500
    for i in range(iter):
        for j in range(traindataArr.shape[0]):
            alpha=4/(1.0+j+i)+0.01
            xi=traindataArr[j]
            yi=trainlabelArr[j]
            # 将(m,)reshape为(1,m)
            xi=xi.reshape(1,xi.shape[0])
            # w*xi的值，即y_hat
            h=sigmoid(np.dot(xi,w))
            # err=y-y_hat
            error=yi-h
            # 更新参数
            w = w + alpha* np.dot(xi.T,error)
    # 返回weights 
    return w

def predict(w,x):
    # 参数列表，输入数据 # return 类别
    wx=np.dot(w.T,x)
    p=sigmoid(wx)
    if p>0.5:
        return 1.0
    else:
        return 0.0

def loadData(filename):
    # 加载数据
    fr=open(filename)
    dataArr=[];labelArr=[]
    for line in fr.readlines():
        # 去掉多余空格，\t分割
        currLine=line.strip().split('\t')
        featList=[]
        for i in range(21):
            featList.append(float(currLine[i]))
        # 特征
        dataArr.append(featList)
        # 标签
        labelArr.append(float(currLine[21]))
    return dataArr,labelArr

def modelTest(testDataArr,testLabelArr,w):
    # return accur
    
    # 填充一列，值为1
    for i in range(len(testDataArr)):
        testDataArr[i].append(1)
    errCnt=0
    for i in range(len(testDataArr)):
        if testLabelArr[i]!=predict(w,testDataArr[i]):
            errCnt+=1
    # return accur
    return 1-errCnt/float(len(testLabelArr))

if __name__=="__main__":
    start=time.time()
    
    # 加载数据
    trainData, trainLabel = loadData("./horseColicTraining.txt")
    testData, testLabel = loadData("./horseColicTest.txt")

    # 开始训练，得到权重w
    w=gradAscent(trainData,trainLabel)

    # 正确率
    accu=modelTest(testData,testLabel,w)

    # 打印
    print("the accur is:",accu)

    # 时间
    end=time.time()
    print("time span:",end-start)


