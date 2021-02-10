


import os
import time
import numpy as np


def file2matrix(filename:str):
    #得到文件内容
    fr=open(filename,'r')
    content=fr.readlines()
    fr.close()
    # 得到文件行数
    fr=open(filename,'r')
    numberOfLines=len(fr.readlines())
    fr.close()
    # matrix，numberOfLines * 3的0阵
    returnMat=np.zeros( (numberOfLines,3) )
    # label, numberOfLines * 1的列向量
    label=[]
    index=0
    """
        readlines方法:一次性读取整个文件；自动将文件内容分析成一个行的列表。
        str.strip() #参数默认，删除头尾空格
        str.lstrip() #删除开头空格,str.rstrip() #删除结尾空格
    """
    for line in content:
        #参数默认，删除头尾空格
        line=line.strip()
        #\t 制表符分割行数据
        listFromLine=line.split('\t')
        #feature
        returnMat[index,:]=listFromLine[0:3]
        #label
        label.append(int(listFromLine[-1]))
        index+=1
    return returnMat,label


def autoNorm(dateSet:np.ndarray):
    # 取dateset的每列的最小值，为np.ndarray类型
    minVals=dateSet.min(0)
    # 取dateset的每列的最大值
    maxVals=dateSet.max(0)
    # 归一化矩阵
    normMat=np.zeros( np.shape(dateSet) )
    # 广播
    normMat=(dateSet-minVals)/(maxVals-minVals)
    return normMat

"""
KNN
"""
def calcDist(x1,x2):
    # 距离度量：欧式距离
    return np.sqrt(np.sum(np.square(x1-x2)))

def getCls(x,dataSet,labels,k):
    # 样本距离
    distList=[0]*len(dataSet)
    # 遍历训练集，计算距离
    for i in range(len(dataSet)):
        # 训练集样本
        x1=dataSet[i]
        # 距离
        dist=calcDist(x,x1)
        # 保存距离
        distList[i]=dist
    # 使用np.argsort()，该函数按值排序，返回索引号序列
    topKList=np.argsort(np.array(distList))[:k]
    # 投票,假设最多有10个类别
    labelList=[0]*10
    for index in topKList:
        tickit=int(labels[index])
        labelList[tickit]+=1
    # 返回索引，即为预测
    return labelList.index(max(labelList)) 



def datingClassTest(datingDataMat,datingLabels,ratio,k):
    """
    测试数据比例: ratio
    """
    # 归一化
    normMat=autoNorm(datingDataMat)
    # 记录错误数
    errorCnt=0
    # 测试数据数量
    total=int(ratio*len(datingLabels))
    for i in range(total):
        print('test %d:%d'%(i,total))
        # 测试向量
        x=normMat[i]
        # 获取预测的label
        y=getCls(x,normMat,datingLabels,k)
        # 统计error
        if y!=datingLabels[i]:
            errorCnt+=1
    # return 正确率
    return 1-(errorCnt/total)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    # 输入数据
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    # 处理
    # 加载数据集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 归一化数据集
    normMat= autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = getCls(inArr,normMat,datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

if __name__ == "__main__":
    start=time.time()

    # 获取训练集
    datingDataMat,datingLabels=file2matrix("./datingTestSet2.txt")
    # 计算测试集的正确率
    accur=datingClassTest(datingDataMat,datingLabels,0.1,10)
    print("accu is:%d"%(accur * 100),"%")

    #classifyPerson() 

    end=time.time()
    # 打印使用的时间
    print("time spand:",end-start)

