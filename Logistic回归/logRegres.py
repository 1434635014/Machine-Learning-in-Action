from numpy import *
# Logistic 回归梯度上升优化算法
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('./data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    # 转换为numpy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (mat(classLabels).transpose() - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit (weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]   # 获取2维矩阵的行，也就是有多少个样本
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

# 随机梯度上升算法
def stocGradAscent0 (dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1 (dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + (alpha * error * dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights

# 示例：从疝气病症预测病马的死亡率
# Logistic回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('./data/horse_colic_train.txt')
    trainingSet = []; trainingLabels = []
    
    #没有用的属性的下标
    index = [2,24,25,26,27]

    for line in frTrain.readlines():
        currLine = line.strip().split(' ')
        lineArr = []
        m = shape(currLine)[0]
        for i in range(m):
            if i in index:
                #没有用的属性直接跳过
                continue
            elif i == 22:
                #下标为22的属性是分类
                #1代表活着，标记设为1
                #2,3分别代表死亡，安乐死，标记设为0
                if currLine[i] == '?':
                    trainingLabels.append(0)
                elif int(currLine[i]) == 1:
                    trainingLabels.append(1)
                else:
                    trainingLabels.append(0)
            else:
                #剩下的是有用数据
                if currLine[i] == '?':
                    #缺失数据首先由0代替
                    lineArr.append(0.0)
                else:
                    lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0

    frTest = open('./data/horse_colic_test.txt')
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split(' ')
        lineArr = []; label = 0.0
        m = shape(currLine)[0]
        for i in range(m):
            if i in index:
                #没有用的属性直接跳过
                continue
            elif i == 22:
                #下标为22的属性是分类
                #1代表活着，标记设为1
                #2,3分别代表死亡，安乐死，标记设为0
                if currLine[i] == '?':
                    label = 0
                elif int(currLine[i]) == 1:
                    label = 1
                else:
                    label = 0
            else:
                #剩下的是有用数据
                if currLine[i] == '?':
                    #缺失数据首先由0代替
                    lineArr.append(0.0)
                else:
                    lineArr.append(float(currLine[i]))

        result = int(classifyVector(array(lineArr), trainWeights))
        trueResult = int(label)
        if (result != trueResult):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is：%f' % errorRate)
    return errorRate
def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is：%f' % (numTests, errorSum / float(numTests)))

