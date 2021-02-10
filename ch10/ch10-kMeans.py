'''
Author: liubai
Date: 2021-02-10
LastEditTime: 2021-02-10
'''

import numpy as np
import matplotlib.pyplot as plt 
import time 

def loadData(filename):
    dataArr=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        # 将列表数据映射为float，
        # 并转为列表形式
        fltLine=list(map(float,curLine))
        dataArr.append(fltLine)
    # 返回np.array()形式的数据集
    return np.array(dataArr)

def distEuc(x,y):
    # 计算两个数据点x，y之间的距离
    # 这里的距离是欧式距离
    return np.sqrt(np.sum(np.power(x-y,2)))

def randCent(dataArr,k):
    # 随机构建K个质心
    n=dataArr.shape[1]
    # k个质心
    centerArr=np.mat(np.zeros((k,n)))
    for j in range(n):
        minVal=dataArr[:,j].min()
        maxVal=dataArr[:,j].max()
        rang=float(maxVal-minVal)
        # # np.random.rand()返回一个或一组服从“0~1”均匀分布的随机样本值。
        # 该随机样本取值范围是[0,1)，不包括1
        # 生成k*1个随机数，范围是[0,1)
        centerArr[:,j]=np.mat(minVal+rang*np.random.rand(k,1))
    return centerArr

def kMeans(dataArr,k,distMeas=distEuc,createCenter=randCent):
    # 每个数据点
    m=dataArr.shape[0]
    # 记录数据点所属的簇编号和最小距离
    clusterAssment=np.mat(np.zeros((m,2)))
    # 初始k个质心
    centerArr=createCenter(dataArr,k)
    # 如果聚簇不发生变化，则聚类结束
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        # 开始遍历每个数据点
        for i in range(m):
            # 计算该样本点距离每个质心的距离
            # 选择最小距离，该样本点属于该簇
            minDis=np.inf;minIdx=-1
            for j in range(k):
                distJ=distMeas(centerArr[j,:],dataArr[i,:])
                if distJ<minDis:
                    minDis=distJ
                    minIdx=j
            if clusterAssment[i,0] != minIdx:
                clusterChanged=True
            clusterAssment[i,:]=minIdx,minDis**2
        # 一边迭代完成,更新质心集
        # print('iter next')
        for cent in range(k):
            # get all the point in this cluster
            ptsInClust = dataArr[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            # 新的质心
            centerArr[cent,:]=np.mean(ptsInClust,axis=0)
    # 当质心点不再发生便阿虎，则返回质心点集、簇编号和最小距离集合
    return centerArr,clusterAssment

def binKmeans(dataArr,k,distMeas=distEuc):
    m=dataArr.shape[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    # 初始簇
    centerArr=np.mean(dataArr,axis=0).tolist()[0]
    # print("centerArr",centerArr)
    centList=[centerArr]
    # 计算每个数据点距离质心的距离综合，即起始误差
    for j in range(m):
        clusterAssment[j,1]=distMeas(np.mat(centerArr),dataArr[j,:])**2
    # 当聚类数小于k时，循环
    while len(centList)<k :
        low_err=np.inf
        for i in range(len(centList)):
            # 获取当前质心的所有数据点
            points_in_cluster= dataArr[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            cluser_index,cluster_assment=kMeans(points_in_cluster,2,distEuc)
            # 误差
            sse_split=np.sum(cluster_assment[:,1])
            sse_no_split=np.sum(cluster_assment[np.nonzero(cluster_assment[:,0].A!=i)[0],1])
            if (sse_no_split+sse_split)<low_err:
                best_split_index=i
                best_new_center=cluser_index
                best_split_ass=cluster_assment.copy()
                # 更新最小误差
                low_err=sse_no_split+sse_split
        # 更改聚类个数
        best_split_ass[np.nonzero(best_split_ass[:,0].A == 1)[0],0] = len(centList) 
        #replace a centroid with two best centroids 
        centList[best_split_index] =  best_new_center[0,:].tolist()[0]
        # 
        centList.append(best_new_center[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == best_split_index)[0],:]=best_split_ass
        #reassign new clusters, and SSE
    return  np.mat(centList),clusterAssment

def plot(dataArr,clusterArr):
    # 画出二维数据点和聚类点
    plt.figure()
    # 数据点
    plt.scatter(dataArr[:,0],dataArr[:,1],c='b',marker='^')
    # 聚类点
    for i in range(clusterArr.shape[0]):
        plt.plot(clusterArr[i,0],clusterArr[i,1],c='r',marker='*')
    plt.show()

if __name__=='__main__':
    start=time.time()
    
    #  加载数据
    print('load data...')
    dataArr=loadData('./testSet.txt')
    
    # 聚类，欧式距离
    print('kmeans...')
    center_point,cluster_assment=kMeans(dataArr,4,distEuc)
    
    end=time.time()
    
    # 聚类结果示意图
    plot(dataArr,center_point)
    
    # time span 
    print('time span:',end-start)
