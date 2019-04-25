from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []

    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return  dataMat, labelMat
# 普通线性回归-------------------------------------------
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("该矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 局部加权线性回归函数--------------------------------
def lwlr (testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 权重单位矩阵
    weights = mat(eye((m)))

    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        # 权重值大小以指数级衰减
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("该矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws

def lwlrTest (testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat
# 岭回归----------------------------------------
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam     # 这里乘以一个单位矩阵和λ

    if linalg.det(xTx) == 0.0:
        print("该矩阵不可逆")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def drawing(xMat, yMat, ws):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    
    # 计算相关系数，越高，表示直线进行建模具有不错的表现
    print('相关系数：')
    print(corrcoef(yHat.T, yMat))

    plt.show()

def drawingLwl(xMat, yMat, yHat):
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:,0].flatten().A[0], s=2, c='red')
    ax.plot(xSort[:, 1], yHat[srtInd])
    plt.show()
    
if __name__ == '__main__':
    xArr, yArr = loadDataSet('../data/ex0.txt')

    # 普通回归
    # ws = standRegres(xArr, yArr)
    # drawing(mat(xArr), mat(yArr), ws)

    # 局部加权回归
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    drawingLwl(mat(xArr), mat(yArr), yHat)
   
