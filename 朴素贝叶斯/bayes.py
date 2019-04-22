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