'''
Author: liubai
Date: 2021-02-05
LastEditTime: 2021-02-05
'''
import numpy as np
import time

def loadData(filename):
    # 加载数据
    dataArr=[];labelArr=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataArr.append([float(lineArr[0]),float(lineArr[1])])
        labelArr.append(float(lineArr[2]))
    return dataArr,labelArr

def selectJrand(i,m):
    # 第i个alpha的下标，m是alpha的总数
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(alpha,H,L):
    # 调整大于或小于alpha的值
    if alpha>H:
        alpha=H
    if alpha<L:
        alpha=L
    return alpha

def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    # selectJ（返回最优的j和Ej）
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
    oS.cache[i] = [1, Ei]

    # 非零E值的行的list列表，所对应的alpha值
    validEcacheList = np.nonzero(oS.cache[:, 0].T)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
            if k == i:
                continue  # don't calc for i, waste of time

            # 求 Ek误差: 预测值-真实值的差
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)

        # 求 Ek误差: 预测值-真实值的差
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    # 计算误差并存入缓存
    """updateEk（计算误差值并存入缓存中。）
    在对alpha值进行优化之后会用到这个值。
    Args:
        oS  optStruct对象
        k   某一列的行号
    """
    # 求 误差: 预测值-真实值的差
    Ek = calcEk(oS, k)
    oS.cache[k] = [1, Ek]

def smoP(dataMatIn, classLabels, C, toler, maxIter,ktup):
    """
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """

    # 创建一个 optStruct 对象
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,ktup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历: 循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    # 循环迭代结束 或者 循环遍历所有alpha后，alphaPairs还是没变化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = np.nonzero((oS.alphas.T > 0) * (oS.alphas.T < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集
    Returns:
        wc  回归系数
    """
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w =w+ np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
  
def kernelTrans(dataArr,dataLine,ktup):
    m,n=np.shape(dataArr)
    kern=np.zeros((m,1))
    if ktup[0]=='lin':
        kern=np.dot(dataArr,dataLine.T)
    elif ktup[0]=='rbf':
        for j in range(m):
            dis=dataArr[j,:]-dataLine
            kern[j]=np.dot(dis,dis.T)
        kern=np.exp(kern/(-1 * ktup[1]**2))
    else:
        print("ktup error")
        return 
    return kern

class optStruct:
    def __init__(self,dataArr,labelArr,C,toler,ktup):
        # 相比无kernel的原SMO，此处多了一个参数
        self.dataMat=np.mat(dataArr)
        self.labelMat=np.mat(labelArr)
        self.C=C
        self.tol=toler
        self.m,self.n=np.shape(dataArr)
        self.alphas=np.zeros((self.m,1))
        self.b=0
        # 是否有效，实际的E值
        self.cache=np.zeros((self.m,2))
        # 新增
        self.K=np.zeros( (self.m,self.m) )
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.dataMat,self.dataMat[i,:],ktup).reshape(self.m)

def calcEk(oS:optStruct,i:int):
    # 给定alpha下计算E
    # W^T
    W=np.multiply(oS.alphas,oS.labelMat)
    # m*1
    # Xi=np.multiply(dataMat,dataMat[i,:].T)
    Xi=oS.K[:,i]
    # 预测值 1*1
    fxi=np.dot(W.T,Xi)+oS.b
    # 计算误差
    Ei=fxi-float(oS.labelMat[i])
    # 返回误差
    return Ei
    
def innerL(i, oS):

    # 求 Ek误差: 预测值-真实值的差
    Ei = calcEk(oS, i)

    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 修改
        eta = 2.0 * oS.K[i,j] * oS.K[i, i] - oS.K[j,j]

        if eta >= 0:
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] =oS.alphas[j]- oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] =oS.alphas[i] + oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # 修改
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j]- oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def modelTest():
    dataArr,labelArr=loadData("./testSetRBF2.txt")
    b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,('rbf',1.3))
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).T
    svIndex=np.nonzero(alphas)[0]
    svS=dataMat[svIndex]
    labelSv=labelMat[svIndex]
    m,n=np.shape(dataMat)
    errCnt=0
    for i in range(m):
        kern=kernelTrans(svS,dataMat[i,:],('rbf',1.3))
        predict=np.dot(kern.T,np.multiply(labelSv,alphas[svIndex]))+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errCnt+=1
    return 1-errCnt/float(m)

if __name__=='__main__':
    start=time.time()

    accuracy=modelTest()

    print("accyr is :",accuracy)
    

    end=time.time()

    print("time span:",end-start)
    






