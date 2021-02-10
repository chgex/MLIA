'''
Author: liubai
Date: 2021-02-01
LastEditTime: 2021-02-01
'''






import math
import time

def calcuEnt(dataset):
    # 计算信息熵
    # 样本个数
    numOfDat=len(dataset)
    # 字典： {标签:出现次数}
    labels={}
    for featVec in dataset:
        #dataset最后一列是label列，即类别
        currLabel=featVec[-1]
        if currLabel not in labels.keys():
            labels[currLabel]=0
        labels[currLabel]+=1
    # 计算熵
    ent=0.0
    for key in labels:
        # 概率
        p=float(labels[key])/numOfDat
        # 熵
        ent-=p*math.log(p,2)
    # 返回该数据集的熵
    return ent

def splitDataSet(dataSet,axis,value):
    # 划分之后的数据集
    retDataSet=[]
    for featVec in dataSet:
        # 如果样本的第axis个特征值为value，则跳过该特征值
        if featVec[axis]==value:
            reducedFeatVec =featVec[0:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeature(dataSet):
    # 样本的特征数,
    # 最后一列是label
    numFeatures = len(dataSet[0]) - 1
    # 数据集信息熵
    baseEntropy = calcuEnt(dataSet)
    # 最优划分下的熵增益值和特征
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        # 保存特征
        featList = [example[i] for example in dataSet]
        # 去重
        uniqueVals = set(featList)
        # 第i个特征划分，数据集的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            # 按照value,进行数据集的划分
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算每个value划分的prob和信息熵
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcuEnt(subDataSet)
        # 判断是否需要为最优划分
        # 信息熵增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 返回bestFeature,bestGain
    return bestFeature



import operator
# 如果数据集已经处理了所有属性，则采用多数表决的方法决定该叶子结点的分类

def majorCnt(classList):
    classCnt={}
    for ticket in classList:
        # 如果不在字典，则需要初始化为0
        if ticket not in classCnt.keys():
            classCnt[ticket]=0
        classCnt[ticket]+=1
    # 按照出现次数排序
    sortedClassCnt=sorted(classCnt.items(),key=operator.itemgetter(1),reverse=True)
    # 如果数据集已经处理了所有属性，则采用多数表决的方法决定该叶子结点的分类
    return sortedClassCnt[0][0]



# 递归构建决策树：原始数据集，基于最好的feature，划分数据集，由于对应的value可能有多个，
# 所以可能存在大于2个分支的数据集划分。
# 第一次划分之后，数据向下传递到分支的下一个结点，在该节点上继续划分，重复如上流程。

# 递归结束条件：
#  1.分支下所有实例都具有相同分类；
#  2.到达叶子结点的数据，属于叶子结点的分类。


def createTree(dataSet,labels):
    # 输入数据集和标签列表，
    # 算法运行中并不需要标签列表，为了给数据一个明确的含义，所以作为一个参数提供。
    # dataset的最后一列是类别
    classList=[example[-1] for example in dataSet]
    # print('start creat node: %d/%d'%(len(dataSet[0]),len(classList)))
    # 边界1：类别都相同，则停止划分,直接返回该类别
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 边界2：如果已经遍历完所有的特征，即没有feature来继续做划分，
    # 则采用投票法，返回出现次数最多的类别。
    if len(dataSet[0])==1:
        return majorCnt(classList)    
    # 继续划分，构建子节点
    # 最优划分的feature
    bestFeat=chooseBestFeature(dataSet) 
    # 构建样本feature列的信息
    bestFeatLabel=labels[bestFeat]
    # 字典形式保存树
    myTree={bestFeatLabel:{}} 
    # 去掉该列，因为在之后的划分中，
    # 该列不在dataSet中了，需要与labels对应
    tmp=labels[:]
    del(tmp[bestFeat])
    subLabels=tmp[:]
    valueList=[example[bestFeat] for example in dataSet]
    uniqueVals=set(valueList)
    for value in uniqueVals:
        # 复制，使操作不影响到原数据
        
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def predict(tree, labels, testVec):
    """
        tree  决策树模型
        labels Feature标签对应的名称
        testVec    测试输入的数据
    Returns: classLabel    分类的结果值，映射label返回名称
    """
    # 根节点对应的key值，
    # 即第一个feature
    firstStr = list(tree.keys())[0]
    # 根节点对应的value值，
    # 即根结点的子树
    secondDict = tree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    # 索引
    featIndex = labels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = predict(valueOfFeat, labels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(tree,filename):
    # 存储决策树模型
    import pickle
    fw=open(filename,'wb')
    pickle.dump(tree,fw)
    fw.close()

def loadTree(filename):
    # 加载决策树模型
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

def modelTest(tree,testDataList, labelList):
    # 计算模型准确率
    # 错误数
    errorCnt = 0
    # 遍历测试集
    for i in range(len(testDataList)):
        if testDataList[i][-1] != predict(tree,labelList,testDataList[i][:-1]):
            errorCnt += 1
    # 返回准确率
    return 1 - errorCnt / len(testDataList)

def loadData(filename):
    fr=open(filename)
    dataList=[example.strip().split('\t') for example in fr.readlines()]
    labelList=['age', 'prescript', 'astigmatic', 'tearRate']
    return dataList,labelList


if __name__=='__main__':
    start = time.time()

    # 训练集
    trainDataList,trainLabelList=loadData('./lenses.txt')
    # 测试集
    testDataList,testLabelList=loadData('./test.txt')
    
    # 创建决策树模型 
    print("start create tree")
    tree=createTree(trainDataList,trainLabelList)
    print("tree is:\n",tree)

    # 准确率
    accur = modelTest(tree,testDataList,testLabelList)
    print('accur is:', accur)

    end = time.time()
    print("time spand: ",end-start)
