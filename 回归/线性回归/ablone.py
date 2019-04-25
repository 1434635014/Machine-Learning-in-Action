# 预测鲍鱼的年龄

import regression

def rssError (yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

if __name__ == '__main__':
    xArr, yArr = regression.loadDataSet('../data/abalone.txt')
    # yHat01 = regression.lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.1)
    # yHat1 = regression.lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
    # yHat10 = regression.lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)
    # # 为了分析预测误差的大小，可以用函数 rssError() 计算出这一指标
    # print(rssError(yArr[0:99], yHat01.T))
    # print(rssError(yArr[0:99], yHat1.T))
    # print(rssError(yArr[0:99], yHat10.T))
    # print('在新数据上的误差：')
    # yHat01 = regression.lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 0.1)
    # yHat1 = regression.lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 1)
    # yHat10 = regression.lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 10)
    # print(rssError(yArr[100:199], yHat01.T))
    # print(rssError(yArr[100:199], yHat1.T))
    # print(rssError(yArr[100:199], yHat10.T))

    # 使用岭回归的方式
    ridgeWeights = regression.ridgeTest(xArr, yArr)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()