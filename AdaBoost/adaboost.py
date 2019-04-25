from numpy import *

def loadSimpData():
    datMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

"""
    将最小错误率 minError 设为+∞
    对数据集中的每一个特征（第一层循环）：
        对每个步长（第二层循环）：
            对每个不等号（第三层循环）：
            建立一棵单层决策树并利用加权数据集对它进行测试
            如果错误率低于 minError ，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树
"""
# 单层决策树生成函数
def stumpClassify (dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones(( shape(dataMatrix)[0], 1 ))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump (dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numStemps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m, 1)))

    minError = inf    # 最小错误率初始值为无穷大

    for i in range(n):
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numStemps

        for j in range(-1, int(numStemps)+1):
            
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0

                weightedError = D.T * errArr
                # print('split: dim %d，thresh %.2f，thresh ineqal：%s，the weighted error is %.3f' % \
                #         (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i                # 错误率最小的一列
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

# 基于单层决策树的AdaBoost训练过程
def abaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D：", D.T)
        alpha = float(0.5 * log(( 1.0-error ) / max(error, 1e-16) ))

        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst：", classEst.T)

        # 未下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        
        aggClassEst += alpha * classEst
        print("aggClassEst：", aggClassEst.T)

        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1) ))
        errorRate = aggErrors.sum() / m
        print("total error：", errorRate, "\n")
        
        if errorRate == 0.0: break
    return weakClassArr

# 测试算法：基于 AdaBoost 的分类
def abaClassify(dataToClass, classifierArray):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArray)):
        classEst = stumpClassify(dataMatrix, classifierArray[i]['dim'], classifierArray[i]['thresh'], classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)

if __name__ == '__main__':
    datMat, classLabels = loadSimpData()
    # D = mat(ones((5, 1)) / 5)
    # buildStump(datMat, classLabels, D)
    classifierArray = abaBoostTrainDS(datMat, classLabels, 9)
    aggClassEst = abaClassify([0, 0], classifierArray)
    print(aggClassEst)