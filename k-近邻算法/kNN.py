from numpy import *
import operator

def createDataSet ():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
# inX：用于分类的输入向量
# dataSet：训练样本集
# labels：标签向量
# k：最近邻居的数目
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# group, labels = createDataSet ()
# print (classify([0, 0], group, labels, 3))

# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLlines = len(arrayOLines)
    returnMat = zeros(( numberOfLlines, 3 ))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 特征值相除
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('data/kNN约会数据样本.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = int(normMat.shape[0])
    numTestVecs = int(m * hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 4)
        # print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the totol error rate is: %f" % (errorCount/float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spen playing video games？"))
    ffMiles = float(input("frequent flier miles earned per year？"))
    iceCream = float(input("liters of ice cream consumed per year？"))
    datingDataMat, datingLabels = file2matrix('data/kNN约会数据样本.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals) / ranges, normMat, datingLabels, 3)
    
    print("You will probably like this persion:", resultList[classifierResult - 1])
