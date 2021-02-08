'''
Author: liubai
Date: 2021-02-08
LastEditTime: 2021-02-08
'''

import numpy as np
import time

def calc_cls_err(dataArr,labelArr,dim,value,rule,D):
    """
    按照第dim列，value值，来划分数据集。
    计算出错误率error，并与样本权重矩阵D相乘
    得到weightsError
    """
    # 转为np.mat
    dataArr,labelArr=np.mat(dataArr),np.mat(labelArr)
    # print("calc:np.shape(dataArr)",np.shape(dataArr))
    # print("calc:np.shape(labelArr)",np.shape(labelArr))
    # 分类误差
    errArr=np.ones( (dataArr.shape[0],1) )
    # 需要的一列；# 标签，用来计算err
    x=dataArr[:,dim];y=labelArr
    # 单层决策树，预测的结果
    retArr=np.ones( (dataArr.shape[0],1) )
    # 标签根据实际情况变化，设置变化规则
    # 按值分类数据集
    if rule == 'lt':
        retArr[x<=value] = -1.0    
    else:
        retArr[x>value] =  -1.0
    # 计算err
    errArr[retArr==labelArr]=0
    # print("cla:retArr",retArr.T)
    # print("cla:labelArr",labelArr.T)
    weightErr=np.dot(D.T,errArr)
    # 返回预测结果，错误率
    return retArr,weightErr

def createStump(dataArr,labelArr,D):
    # m*n;m*1
    dataMatrix = np.mat(dataArr); labelMat = np.mat(labelArr).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}; bestPredict = np.mat(np.zeros((m,1)))
    minError = np.inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for rule in ['lt', 'gt']: #go over less than and greater than
                value = (rangeMin + float(j) * stepSize)
                predictedVals,weightedError=calc_cls_err(dataMatrix,labelMat,i,value,rule,D)
                if weightedError < minError:
                    minError = weightedError
                    bestPredict = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['value'] = value
                    bestStump['rule'] = rule
    return bestStump,minError,bestPredict

def createBoostingTree(dataArr,labelArr,iter=40):
    # m*n,m*1
    dataArr=np.mat(dataArr);labelArr=np.mat(labelArr)
    m,n=np.shape(dataArr)
    finalPredict=np.zeros( (m,1) )
    # 初始化样本权重D
    D=np.ones((m,1))/m
    # tree
    boostTree=[]
    # 迭代
    for i in range(iter):
        bestSingleTree,err,predict=createStump(dataArr,labelArr,D)
        alpha=0.5*float( np.log( (1.0-err)/max(err,1e-16) ) )
        # print('alpha',alpha)
        bestSingleTree['alpha']=alpha
        boostTree.append(bestSingleTree)
        expon=np.multiply(-1 * alpha * labelArr.T,predict)   
        # print("expon",expon)
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()
        # print("D",D)
        finalPredict=finalPredict+alpha*predict
        finalErr = np.multiply(np.sign(finalPredict) != np.mat(labelArr).T,np.ones((m,1)))
        errRate=finalErr.sum()/m
        print("total error:",errRate)
        if errRate==0.0: break 
    return boostTree

def loadDataSet(fileName):      
    # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,value,rule):
    # 单结点数据分类
    #just classify the data
    retArray = np.ones( (dataMatrix.shape[0],1) )
    if rule == 'lt':
        retArray[dataMatrix[:,dimen] <= value] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > value] = -1.0
    return retArray

def classify(dataArr,label,clsArr):
    # 集成单节点数据分类结果
    dataMatrix = np.mat(dataArr)
    m = np.shape(dataMatrix)[0]
    aggCls = np.mat(np.zeros((m,1)))
    for i in range(len(clsArr)):
        classEst = stumpClassify(dataMatrix,clsArr[i]['dim'],\
                                 clsArr[i]['value'],\
                                 clsArr[i]['rule'])
                                 #call stump classify
        aggCls += clsArr[i]['alpha']*classEst
    # 返回预测结果矩阵
    return np.sign(aggCls)

def modelTest(dataArr,labelArr,clsArr):
    # return caauracy
    predictArr=classify(dataArr,labelArr,clsArr)
    errArr=np.ones( (len(labelArr),1) )
    # 分类错误的样本个数
    errCnt=errArr[predictArr!=np.mat(labelArr).T].sum()
    # 返回准确率
    return 1-errCnt/float(len(labelArr))


import matplotlib.pyplot as plt
def plotROC(dataArr, labelArr,clsArr):  
    predict= classify(dataArr,labelArr,clsArr).T
    cur = (1.0,1.0) 
    #cursor
    ySum = 0.0 
    #variable to calculate AUC
    numPosClas = np.sum(np.array(labelArr)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(labelArr)-numPosClas)
    sortedIndicies = predict.argsort()
    # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if labelArr[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

if __name__=='__main__':
    start=time.time()

    # 加载数据
    traiDataArr,trainLabelArr=loadDataSet("./horseColicTraining2.txt")
    
    # 训练分类器
    # 20个弱分类器
    clsArr=createBoostingTree(traiDataArr,trainLabelArr,20)

    # 分类器性能
    testDataArr,testLabelArr=loadDataSet("./horseColicTest2.txt")
    accuracy=modelTest(testDataArr,testLabelArr,clsArr) 
    # print accur
    print("The accuracy is:",accuracy)
    
    end=time.time()

    # plot ROC曲线
    plotROC(testDataArr,testLabelArr,clsArr)
    
    # time span
    print("time span:",end-start)

    

