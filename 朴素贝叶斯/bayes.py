from numpy import *

def loadDataSet():
    postingList = [["my", "dog", "has", "flea", "problems", "help", "please"],
                    ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
                    ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
                    ["stop", "posting", "stupid", "worthless", "garbage"],
                    ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
                    ["quit", "buying", "worthless", "dog", "food", "stupid"]]
    classVec = [0, 1, 0, 1, 0, 1]   # 1：侮辱性文字，0：正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])                      # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)        # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("这单词: %s 不在我的词汇里！" % word)
    return returnVec
# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)        # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        # else: print("这单词: %s 不在我的词汇里！" % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategrory):
    numTrainDocs = len(trainMatrix) # 要训练的向量个数
    numWords = len(trainMatrix[0])  # 要训练的向量长度，也就是总单词个数
    pAbusive = sum(trainCategrory) / float(numTrainDocs)    # 侮辱性的总概率P(A)，A为侮辱性概率
    # 这里为了避免乘积之后全部为0，将所有词的出现数初始化为1，并将分母初始化为2
    # p0Num = zeros(numWords); p1Num = zeros(numWords)        # 创建向量
    # p0Denom = 0.0; p1Denom = 0.0                            # （侮辱性，非侮辱性）各自总单词个数
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0 
    for i in range(numTrainDocs):
        if trainCategrory[i] == 1:  # 侮辱性
            p1Num += trainMatrix[i] # 向量相加
            p1Denom += sum(trainMatrix[i])  # 单词个数相加
        else:                       # 非侮辱性
            p0Num += trainMatrix[i] # 向量相加
            p0Denom += sum(trainMatrix[i])  # 单词个数相加
    # 这里为了避免大量小的数字乘积四舍五入后为0，所以使用 ln 来算
    # p1Vect = p1Num/p1Denom          # 向量相除，得到侮辱性各个单词的概率，P(w1,w2,...|A)
    # p0Vect = p0Num/p0Denom          # 向量相除，得到非侮辱性各个单词的概率
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pAbusive):
    p1 = sum(vec2Classify * p1Vec) + log(pAbusive)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMatrix = []
    for post in listOPosts:
        trainMatrix.append(setOfWords2Vec(myVocabList, post))
    p0V, p1V, pAb = trainNB0(trainMatrix, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as：', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as：', classifyNB(thisDoc, p0V, p1V, pAb))

# 文本解析
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if (len(tok) > 2)]

# 垃圾邮件测试函数
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)         # 垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)         # 正常邮件
    trainingSet = list(range(50)); testSet = []
    vocabList = createVocabList(docList)
    # 随机抽取十个测试邮件数据
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    # 创建训练集和对应的分类
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSapm = trainNB0(trainMat, trainClasses)
    # 测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[int(docIndex)])
        if classifyNB(wordVector, p0V, p1V, pSapm) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is：', str(float(errorCount) / len(testSet)))


# 使用朴素贝叶斯分类器从个人广告中获取区域倾向

# 收集数据：导入 RSS 源
# import feedparser
# ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
# sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
# ny['entries']
# len(ny['entries'])

# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]
def localWords(feed1, feed0):
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen)); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is：', str(float(errorCount) / len(testSet)))
    return vocabList, p0V, p1V