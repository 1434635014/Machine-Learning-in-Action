from numpy import *

# K-均值聚类支持函数
def loadDataSet (fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat
# 计算欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))
# 生成随机的k个簇质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# K-均值聚类算法
def kMeans (dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))    # 聚类评估
    centroids = createCent(dataSet, k)
    clusterChanged = True
    
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1

            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
             #此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代）
            if clusterAssment[i, 0] != minIndex: clusterChanged =  True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 更新质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment
        
# 二分 K-均值算法
# 将所有点看成一个簇
# 当簇数目小于k时
#     对于每一个簇
#         计算总误差
#         在给定的簇上面进行K-均值聚类（k=2）
#         计算将该簇一分为二之后的总误差
#     选择使得误差最小的那个簇进行划分操作
def biKmeans (dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))             # 聚类评估
    centroid0 = mean(dataSet, axis=0).tolist()[0]   # 每一列的均值
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    
    print(clusterAssment)
    # 如果分类数小于k
    while (len(centList) < k):
        lowestSSE = inf     # 最小的分类误差
        
        for i in range(len(centList)):
            # 尝试划分每一簇
            ptsIncurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 切分两份，得到误差
            centroidMat, splitClustAss = kMeans(ptsIncurrCluster, 2, distMeas)
            # 计算误差总和
            sseSplit = sum(splitClustAss[:, 1])
            # 剩余数据集的误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and not Split：", sseSplit, sseNotSplit)

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i                     # 待划分的簇
                bestNewCents = centroidMat              # 划分的簇的质心
                bestClustAss = splitClustAss.copy()     # 划分的簇的误差数组
                lowestSSE = sseSplit + sseNotSplit      # 划分簇的总误差
        # 开始切分
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)          # 把误差函数为 1的下标 新增区域的下标（也就是被划分出来的） 
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit        # 0 的下标属于 改成将要被划分的区域下标（也就是没被划分出去的）
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss     # 更新原有的被划分的误差 值

        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]                      # 改变当前被切分的 质心（划分成0的那一块）
        centList.append(bestNewCents[1, :].tolist()[0])                                             # 新增质心为 （划分成1的那一块）
    return mat(centList), clusterAssment

#distance calc function：结合两个点经纬度（用角度做单位），返回地球表面两点之间距离
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b) * 6371.0
# 球面距离计算及簇绘图函数

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('./data/places.txt').readlines():
        lineArr  = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])

    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('./data/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ == '__main__':
    # 普通的k-均值
    # datMat = mat(loadDataSet('./data/testSet.txt'))
    # myCentroids, clustAssing = kMeans(datMat, 4)
    # print(myCentroids)
    # print(clustAssing)

    # 二分的k-均值
    # datMat3 = mat(loadDataSet('./data/testSet2.txt'))
    # centList, myNewAssments = biKmeans(datMat3, 3)
    # print(centList)
    # print(myNewAssments)

    # 对地图上的点进行聚类
    clusterClubs(6)