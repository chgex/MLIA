'''
Author: liubai
Date: 2021-02-10
LastEditTime: 2021-02-10
'''

import numpy as np
import time  


def loadData(filename):
    # 加载数据
    dataArr=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        # 将数据映射为浮点型
        fltLine=list(map(float,curLine))
        dataArr.append(fltLine)
    return np.array(dataArr)

def binSplitDataSet(dataArr,dim,value):
    # 按第dim列的value值，将数据集划分为两部分
    # index
    idx1=np.nonzero(dataArr[:,dim]>value)[0]
    arr1=dataArr[idx1,:]
    # print("arr1.shape",arr1.shape)
    # index
    idx2=np.nonzero(dataArr[:,dim]<=value)[0]
    arr2=dataArr[idx2,:]
    # print("arr2.shape",arr2.shape)
    return arr1,arr2

# 回归树

def regLeaf(dataArr):
    # 均值
    return np.mean(dataArr[:,-1])

def regErr(dataArr):
    # 总的方差
    return np.var(dataArr[:,-1]) * dataArr.shape[0] 


def chooseBestSplit(dataArr,leafType=regLeaf,errType=regErr,opt=(1,4)):
    # 如果找到好的切分数据集的方式，则返回特征编号和特征值，
    # 如果找不到好的二元切分方式，则返回None，并产生一个叶节点，
    # 叶节点的值，也返回None
    # opt(tolErr,tolNum)为用户指定的，用于控制参数的停止时机,
    # tolErr是容许的误差下降值,
    # tolNum是切分的最少样本数。
    tolErr,tolNum=opt
    # 如果待划分的特征都相同，则返回None，并生成叶节点(叶节点返回的是均值)
    lst=dataArr[:,-1].T.tolist()
    if len(set(lst))==1:
        print("leaf node")
        return None,leafType(dataArr)
    m,n=dataArr.shape
    E=errType(dataArr)
    # 最优划分方式下的方差，特征列编号及其特征值
    bestErr=np.inf;bestIndex=0;bestValue=0
    for idx in range(n-1):
        for val in set(dataArr[:,idx]):
            arr1,arr2=binSplitDataSet(dataArr,idx,val)
            if len(arr1)<tolNum or len(arr2)<tolNum:
                continue
            # 两个结点的总方差
            err=errType(arr1)+errType(arr2)
            if err<bestErr:
                bestIndex=idx
                bestValue=val
                bestErr=err
    # 如果误差减少不大，则提前退出
    if E-bestErr <tolErr:
        return None,leafType(dataArr)
    # 继续建树
    # print("arr1,arr2")
    arr1,arr2=binSplitDataSet(dataArr,bestIndex,bestValue)
    # 如果划分出的数据集样本个数少于阈值，则返回叶结点
    if len(arr1)<tolNum or len(arr2)<tolNum:
        return None,leafType(dataArr)
    # 返回特征编号，特征值
    # print('bestIndex,bestValue')
    return bestIndex,bestValue

# 构建树
def createTree(dataArr,leafType=regLeaf,errType=regErr,opt=(1,4)):
    # 建树，使用字典存储树
    idx,val=chooseBestSplit(dataArr,leafType,errType,opt)
    # 叶节点
    if idx==None:
        return val
    # 树
    # print('tree')
    tree={};
    tree['idx']=idx;tree['val']=val
    lArr,rArr=binSplitDataSet(dataArr,idx,val)
    # 分支
    # print('branch')
    tree['left'] =createTree(lArr,leafType,errType,opt)
    tree['right']=createTree(rArr,leafType,errType,opt)
    # 返回回归树
    return tree

# 剪枝
def isTree(obj):
    # 树使用字典存储，
    # 所以类型是dict的就是子树(或树)
    return (type(obj).__name__=='dict')

def getMean(tree):
    # 计算两个子树的平均总方差
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    return (tree['left']+tree['right'])/2.0

def prune(tree,testArr):
    # 剪枝
    # 没有测试数据了，则对树做塌陷处理
    if testArr.shape[0]==0:
        return getMean(tree)
    # 分支，则使用回归树划分测试集
    if isTree(tree['left']) or isTree(tree['right']):
        lArr,rArr=binSplitDataSet(testArr,tree['idx'],tree['val'])
    # 左分支
    if isTree(tree['left']):
        tree['left']= prune(tree['left'],lArr)
    # 右分支
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rArr)
    # 叶结点
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 计算总的方差
        # 不合并的总方差
        lArr,rArr=binSplitDataSet(testArr,tree['idx'],tree['val'])
        lerr=np.power(lArr[:,-1] - tree['left'],2)
        rerr=np.power(rArr[:,-1] - tree['right'],2)
        errNoMerge=np.sum(lerr) + np.sum(rerr)
        # 合并的总方差
        treeMean=(tree['left']+tree['right'])/2.0
        errMerge=np.sum(np.power(testArr[:,-1] - treeMean,2))
        # 比较
        if errMerge<errNoMerge:
            print('merge')
            # 合并分支
            # 返回两个分支的总方差之和
            return treeMean
        else:
            # 保留原分支
            return tree
    else:
        return tree
    
#  模型树
def linear(dataArr):
    m,n=dataArr.shape
    # 初始化X,Y
    X=np.ones((m,n));Y=np.ones((m,1))
    # 赋值
    X[:,1:n]=dataArr[:,0:n-1];Y=dataArr[:,-1]
    X=np.mat(X);Y=Y.reshape(m,1)
    # print(X.shape,Y.shape)
    xTx=np.dot(X.T,X)
    # print("xTx.shape",xTx.shape)
    # np.linalg.det(X)表示计算矩阵X的行列式
    if np.linalg.det(xTx) == 0.0:
        # 说明不可逆,报错并返回
        print("This matrix cannot do inverse")
        # 求伪逆
        xTx_I=np.linalg.pinv(xTx)
    else:
        # 求逆
        xTx_I=xTx.I
    t=np.dot(X.T,Y)
    # ws = xTx_I*(X.T*Y)
    ws = np.dot(xTx_I,t)
    return ws,X,Y

def modelLeaf(dataArr):
    ws,X,Y=linear(dataArr)
    return ws

def modelErr(dataArr):
    # 在给定的数据集上计算误差
    # 权重矩阵，自变量，因变量
    ws,X,Y=linear(dataArr)
    # 预测值
    yHat=X*ws
    # 返回平方误差
    return np.sum(np.power(Y-yHat,2))

# 使用corrcoef函数分析模型

def regTreeEval(model, inDat):
    # 对叶结点数据的预测
    # 回归树模型
    return float(model)

def modelTreeEval(model, inDat):
    # 对叶节点的预测
    # 模型树
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    # 对于给定的tree，输入值，模型类型，
    # 该函数返回一个预测值
    if not isTree(tree): 
        return modelEval(tree, inData)
    if inData[tree['idx']] > tree['val']:
        if isTree(tree['left']): 
            return treeForeCast(tree['left'], inData, modelEval)
        else: 
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): 
            return treeForeCast(tree['right'], inData, modelEval)
        else: 
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, model=regTreeEval):
    # 返回预测值向量
    m=len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), model)
    return yHat

# test
def modelTest(trainArr,testArr,model):
    # return R^2
    if model=='regTree':
        print('model train...')
        tree=createTree(trainArr,opt=(1,20))
        # 估计值
        yhat=createForeCast(tree,testArr[:,0])
    else:
        print('model train...')
        tree=createTree(trainArr,modelLeaf,modelErr,(1,20))
        # 估计值
        yhat=createForeCast(tree,testArr[:,0],modelTreeEval)
    # R^2   
    return np.corrcoef(yhat,testArr[:,1],rowvar=0)[0,1]

if __name__=='__main__':
    
    start=time.time()

    # load dataArr
    print('load dataArr...')
    trainArr=loadData('./bikeSpeedVsIq_train.txt')
    testArr=loadData('./bikeSpeedVsIq_test.txt')

    # model test
    # 回归树
    corrcoef=modelTest(trainArr,testArr,'regTree')
    print('regTree corrcoef is: ',corrcoef)
    # 模型树
    corr=modelTest(trainArr,testArr,'modelTree')
    print('modelTree corrcoef is:',corr)

    end=time.time()
    # time span
    print('time span is:',end-start)