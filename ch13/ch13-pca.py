'''
Author: liubai
Date: 2021-02-14
LastEditTime: 2021-02-14
'''
import numpy as np 
import matplotlib.pyplot as plt 

# 加载数据
def loadData0(filename,delim='\t'):
    fr=open(filename)
    # 构建矩阵
    stringArr=[]
    for line in fr.readlines():
        curLine=line.strip().split(delim)
        stringArr.append(curLine)
    dataArr=[]
    for line in stringArr:
        dataArr.append(list(map(float,line)))
    return np.mat(dataArr)

# 加载数据
def loadData(filename,delim='\t'):
    fr=open(filename)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    dataArr=[list(map(float,line)) for line in stringArr]
    return np.array(dataArr)

# 主成分分析
def pca(dataArr,topN=9999999):
    """
    dataArr为数据集
    topN参数可选，表示应用的N个特征，
    或原数据中全部特征
    """
    # 首先减去原始数据集的平均值
    # 然后计算协方差矩阵及其特征值
    # 使用argsort()函数对特征值排序，得到其索引，
    # 选择前TOPN个特征向量构成转换矩阵
    # 使用转换矩阵，将数据转换到新空间中，返回转换后的数据集，即为降维数据集。
    # 1 去除平均值
    meanVal=np.mean(dataArr,axis=0)
    newArr=dataArr-meanVal
    # 协方差矩阵
    covMat=np.cov(np.mat(newArr),rowvar=0)
    # 计算特征值和特征向量
    eigVal,eigVec=np.linalg.eig(np.mat(covMat))
    # 按照特征值从小到大进行排序
    index=np.argsort(eigVal)
    # 只取N个最大的特征值
    index=index[:-(topN+1):-1]
    # N个特征值对应的特征向量
    vecArr=eigVec[:,index]
    # 低维度数据集
    lowDataMat=np.mat(newArr)*vecArr
    # 新数据集
    retDataArr=(lowDataMat*vecArr.T)+meanVal
    return lowDataMat,retDataArr

def load_clear_Data():
    import pandas as pd
    # 文件中第一行不是索引 
    data=pd.read_csv('./secom.data',sep=' ',header=None,index_col=None)
    # 计算每一列的均值，并填充NaN值
    for i in range(data.shape[1]):
        # 第i列
        data[i]=data[i].fillna(data[i].mean(),)    
    # 转为矩阵
    dataArr=np.array(data)
    return dataArr

def plot_var(dataMat):
    # 方差与主成分个数的关系
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals 
    # 减去均值
    covMat = np.cov(meanRemoved, rowvar=0)
    # 特征值与特征向量
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    # 排序，取N个最大的特征值及其特征向量
    eigValInd = np.argsort(eigVals)
    # reverse
    eigValInd = eigValInd[::-1]
    sortedEigVals = eigVals[eigValInd]
    # 记录方差变化
    total = np.sum(sortedEigVals)
    varPercentage = sortedEigVals/total*100
    # 画出方差变化与特征个数的关系图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 21), varPercentage[:20], marker='^')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.show()

def plot_data(dataArr,reconMat):
    # 效果图
    fig=plt.figure()
    # 子图
    ax=fig.add_subplot(111)
    ax.scatter(dataArr[:,0],dataArr[:,1],marker='^',s=90,c='b')
    ax.scatter(np.array(reconMat[:,0]),np.array(reconMat[:,1]),marker='o',s=50,c='r')
    plt.show()

if __name__=='__main__':
    
    # 加载数据
    # testSet.txt之后两个维度
    dataArr=loadData('./testSet.txt')
    
    # 将为前数据，降维后的数据
    lowData,reconMat=pca(dataArr,1)
    # 降维前后变化
    plot_data(dataArr,reconMat)
    
    # 加载数据
    dataArr2=load_clear_Data()
    # 主特征个数与方差的关系
    plot_var(dataArr2)

    

    


