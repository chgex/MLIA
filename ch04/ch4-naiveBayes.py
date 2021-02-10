'''
Author: liubai
Date: 2021-02-02
LastEditTime: 2021-02-02
'''
import time
import numpy as np 

def createVocabList(dataSet):
    # 所有单词的集合
    vocabList=[]
    # 去重
    vocabSet=set([])
    for textLine in dataSet:
        # 并集
        vocabSet=vocabSet | set(textLine)
    # 以列表形式返回
    vocabList=list(vocabSet);vocabList.sort()
    return vocabList

def wordBagVec(vocabList,textLine):
    # 词袋向量
    retVec=[0]*len(vocabList)
    for word in textLine:
        if word in vocabList:
            retVec[vocabList.index(word)]=1
    return retVec

def getProb(trainData, trainCategory):
    # 参数：文本单词矩阵，文本类型
    # 样本个数
    numTrainDocs = len(trainData)
    # 词袋的单词数
    numWords = len(trainData[0])
    # 样本类别
    classNum=2
    # 类被为1，先验概率
    Py = np.sum(trainCategory) / float(numTrainDocs)
    # 条件概率，两个类别，对应两个向量
    Px_y=np.ones( (classNum,numWords) )
    
    # 整个数据集，不同类别下，单词出现的总数
    p_sum=np.zeros( (classNum,1) )
    p_sum+=2.0
    # 矩阵：不同类别下，每个单词出现的概率
    pVec=np.zeros( (classNum,numWords) )
    
    # 遍历每一个文本
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            Px_y[1] += trainData[i] 
            # 对向量中的所有元素进行求和，
            # 即计算类别为1的文件中，出现的单词的总数
            p_sum[1] += np.sum(trainData[i])
        else:
            # 类别为0
            Px_y[0] += trainData[i]
            p_sum[0] += np.sum(trainData[i])
    # 每个类别下，每个单词出现的概率
    # 如，类别0，正常文档，[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    pVec=np.log(Px_y / p_sum)
    # 返回先验概率，类别为1的全概率
    return pVec, Py   

def classify(textVec, pVec, pClass1):
    # 根据概率，进行分类
    # 两个类别
    # P(w|c1) * P(c1)，即贝叶斯准则的分子
    p1 = np.sum(textVec * pVec[1]) + np.log(pClass1) 
    # P(w|c0) * P(c0)
    p0 = np.sum(textVec * pVec[0]) + np.log(1.0 - pClass1) 
    if p1 > p0:
        return 1
    else:
        return 0

def textSplit(text):
    # 将文本划分为单词组成的列表
    import re
    retList=[]
    reg = re.compile('\W')
    wordList = reg.split(text)
    retList=[word.lower() for word in wordList if len(word)>2]
    return retList

def loadData():
    # 加载数据
    # 数据集
    dataList=[]
    # 标签集
    labelList=[]
    # 训练集 ：测试集合,划分比例：rate
    rate=0.7
    # 一共选择23*2个邮件
    for i in range(1, 23):
        # spam
        wordList = textSplit(open('./email/spam/%d.txt' % i).read())
        dataList.append(wordList)
        labelList.append(1)
        # ham
        wordList = textSplit(open('./email/ham/%d.txt' % i).read())
        dataList.append(wordList)
        labelList.append(0)
    # train
    trainDataArr=dataList[:int(rate*len(dataList))]
    trainLabelArr=labelList[:int(rate*len(dataList))]
    # test
    testDataArr=dataList[int(rate*len(dataList)):]
    testLabelArr=labelList[int(rate*len(dataList)):]

    return trainDataArr,trainLabelArr,testDataArr,testLabelArr

def data2Mat(trainDataArr,trainLabelArr):
    # 文本数据转为词向量矩阵
    # 创建词汇表    
    vocabList = createVocabList(trainDataArr)
    # 文本数据转为词向量矩阵
    dataMat = []
    labels = []
    for i in range(len(trainDataArr)):
        dataMat.append(wordBagVec(vocabList,trainDataArr[i] ))
        labels.append(trainLabelArr[i])
    # 返回训练数据的词向量，类别标签，词袋
    return dataMat,labels,vocabList

def modelTest(pV, pSpam, testDataArr, testLabelArr, vocabList):
    # 模型测试，返回正确率
    errCnt = 0
    for i in range(len(testDataArr)):
        wordVector = wordBagVec(vocabList,testDataArr[i] )
        if classify(np.array(wordVector), pV, pSpam) != testLabelArr[i]:
            errCnt += 1
    print ('errCnt:',errCnt,',','total: ',len(testLabelArr))
    return 1.0-float(errCnt)/len(testLabelArr)

if __name__=="__main__":
    start=time.time()
    
    # 加载数据
    # 获取训练集，测试集
    trainDataArr,trainLabelArr,testDataArr,testLabelArr=loadData()
    
    # 训练用的文本数据，转为词向量矩阵
    trainDataMat,trainLabels ,vocabList= data2Mat(trainDataArr,trainLabelArr)
    # 计算概率
    pV, pSpam = getProb(np.array(trainDataMat), np.array(trainLabels))

    # 模型测试
    accur= modelTest(pV, pSpam, testDataArr, testLabelArr, vocabList)
    print ('the accur is:',accur)

    # 时间花费
    end=time.time()
    print('time span:',end-start) 