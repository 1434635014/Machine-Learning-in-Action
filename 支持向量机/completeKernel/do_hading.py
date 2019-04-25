# -*- coding: utf-8 -*-
"""
    识别手写数字
"""

import svmMLiA
#import numpy as np
import time 

start=time.clock()
svmMLiA.testDigits(('rbf',10))

end=time.clock()
total_time=end-start
print("Time For Run CompleteSMOWithKernel:"+str(total_time))