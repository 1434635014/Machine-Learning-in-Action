from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['不浮出水面是否可以生存', '是否有脚蹼']
    return dataSet, labels

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    newDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]         # 截取前半部分
            reducedFeatVec.extend(featVec[axis+1:]) # 截取后半部分并拼接在前半部分后面
            newDataSet.append(reducedFeatVec)
    return newDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestEntropy = 0.0; baseFeature = 0
    for i in range(numFeatures):
        featList = [exp[i] for exp in dataSet]      # 获取一列数据
        uniqueVals = set(featList)                  # 去重
        newEntropy = 0.0                            # 信息熵
        for value in uniqueVals:
            # 按照该方式分割数据
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 如果该分割方式比上次最好的分割方式信息熵大，则替换最大的信息熵
        if (infoGain > bestEntropy):
            bestEntropy = infoGain
            baseFeature = i
    return baseFeature

# 多数表决的方法决定该叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树的函数代码
def createTree(dataSet, labels):
    classList = [exp[-1] for exp in dataSet]        # 获取所有的分类
    # 如果只剩下一分类了，则直接返回改分类名称
    if classList.count(classList[0] == len(classList)):
        return classList[0]
    # 如果只剩下一列问题项了，但分类存在多个，则返回多数表决的方法决定该叶子节点的分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最好的分割方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = { bestFeatLabel: {} }  # 以问题项项为属性名的树节点对象
    newLabels = labels[:]   
    del(newLabels[bestFeat])           # 删除labels中的问题项
    featValues = [exp[bestFeat] for exp in dataSet] # 获取该问题项的一列结果
    uniqueVals = set(featValues)                    # 去重
    for value in uniqueVals:
        subLabels = newLabels[:]                       # 复制出一个新的问题项数组，以免修改到 labels本身
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 获取下标
    classLabel = ''
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

# 使用pickle模块存储决策树
def saveTree(inputTree, filename = './saveTree/trees.txt'):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
def getTree(filename = './saveTree/trees.txt'):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


# 示例：使用ID3决策树预测隐形眼镜类型
fr = open('./data/lenses.txt')
lenses = [line.strip().split('\t') for line in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
import treePlotter
treePlotter.createPlot(lensesTree)