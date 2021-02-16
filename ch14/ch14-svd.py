'''
Author: liubai
Date: 2021-02-16
LastEditTime: 2021-02-16
'''

import numpy as np
 

def eculSim(x,y):
    # 相似度计算值之欧式距离
    return 1.0/(1.0+ np.linalg.norm(x-y))

def pearSim(x,y):
    # 相似度计算值之皮尔逊距离
    if len(x)<3:
        return 1.0
    # -1到1转换到0到1
    # corr返回四个值，只需要第二个值
    return 0.5 + 0.5*np.corrcoef(x,y,rowvar=0)[0][1] 

def cosSim(x,y):
    # 相似度计算值之余弦相似度
    num=float(x.T*y)
    denom=np.linalg.norm(x)*np.linalg.norm(y)
    return 0.5+0.5*(num/denom)
    
def loadExData():
    return[[4, 4, 0, 2, 2],
        [4, 0, 0, 3, 3],
        [4, 0, 0, 1, 1],
        [1, 1, 1, 2, 0],
        [2, 2, 2, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 1, 0, 0]]

# 给定相似度计算方法，计算用户对某一物品的评分值
def stand_estimate(dataArr,user,simMeans,item):
    # item为物品编号
    n=dataArr.shape[1]
    sim_total=0.0
    rate_total=0.0
    # n个物品
    for j in range(n):
        user_rate=dataArr[user,j]
        if user_rate==0.0:
            continue
        # 寻找两个用户都评分的物品
        ids=np.nonzero(np.logical_and(dataArr[:,item].A>0,dataArr[:,j].A>0))[0]
        if len(ids)==0:
            sim_rate=0.0
        else:
            sim_rate=simMeans(dataArr[ids,item],dataArr[ids,j])
        # 相似度加和
        sim_total+=sim_rate
        rate_total+=sim_rate*user_rate
    if sim_total==0:
        return 0
    # 归一化相似度评分，这些评分用于对预测值排序
    return rate_total/sim_total

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=stand_estimate):
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1] # find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def svd_estimate(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0; ratSimTotal = 0.0
    # SVD分解
    U,Sigma,VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4]) 
    # 取前90的奇异值
    xformedItems=dataMat.T * U[:,:4] * Sig4.I  
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: 
            continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal/simTotal
    
if __name__=='__main__':

    dataArr=np.mat(loadExData())
    # 预测评分
    est1=recommend(dataArr,1,estMethod=stand_estimate)
    est2=recommend(dataArr,1,estMethod=svd_estimate)
    
    # 打印预测结果
    print('stand estimate is:',est1)
    print(' svd  estimate is:',est2)
    