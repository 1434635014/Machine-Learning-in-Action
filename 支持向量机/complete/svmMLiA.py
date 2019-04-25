# 完整版

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:53:06 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import  numpy as np
import matplotlib.pyplot as plt

'''#######********************************
Non-Kernel VErsions below
'''#######********************************
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')     #delete ' ',\n  split with '\t'
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0];print("the length of validEcacheList",len(validEcacheList))
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
	
     
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

#by myself
def calculateW(dataArr,labelArr,alpha):
	Mw = np.matrix(np.zeros(np.shape(dataArr)[1]))
	for i in range (np.shape(alpha)[0]):
		if alpha[i]>0:
			Mw += np.multiply(labelArr[i]*alpha[i],dataArr[i])
	w = Mw.T.tolist()
	return w

#by myself
def drawing(dataArr,labelArr,alpha,w,b):
	n = np.shape(labelArr)[0] 
	xcord1 = []; ycord1 = []   
	xcord2 = []; ycord2 = []
	xcord3 = []; ycord3 = []
	xcord4 = []; ycord4 = []
	for i in range(n):
		if int(labelArr[i])== 1:
			if alpha[i]>0:
				xcord3.append(dataArr[i][0]); ycord3.append(dataArr[i][1])
			else:
				xcord1.append(dataArr[i][0]); ycord1.append(dataArr[i][1])
		else:
			if alpha[i]>0:
				xcord4.append(dataArr[i][0]); ycord4.append(dataArr[i][1])
			else:
				xcord2.append(dataArr[i][0]); ycord2.append(dataArr[i][1]) 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=40, c='yellow', marker='s',label='class 1')
	ax.scatter(xcord2, ycord2, s=40, c='green',label='class -1')
	ax.scatter(xcord3, ycord3, s=40, c='red',marker='s',label='SV')		
	ax.scatter(xcord4, ycord4, s=40, c='red',label='SV')	

	ax.legend(loc='best')

	x = np.arange(2.7, 6.6, 0.1)
	y1 = (b+(calculateW(dataArr,labelArr,alpha)[0][0])*x)/(-calculateW(dataArr,labelArr,alpha)[1][0])
	y = np.mat(y1).T
	ax.plot(x, y,'-')
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.savefig('SMOComplete.eps',dpi=2000)
	plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('../data/testSet.txt')
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40)
    w = calculateW(dataMat, labelMat, alphas)
    drawing(dataMat, labelMat, alphas, w, b)
