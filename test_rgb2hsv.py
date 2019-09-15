# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:54:48 2018

@author: louis
"""

import matplotlib
import matplotlib.pyplot as plt
from extractImagePartPortion import splitImage
import os
import numpy as np

currentPath = os.path.dirname(os.path.abspath(__file__))
imPath = os.path.join(currentPath, '..', 'data', 'gourd_c1818', 'Images', '03')
imName = '03_00000466.jpg'
imName = '03_00001466.jpg'
#    imName = '03_00000466.jpg'

#    imName = '03_00000654.jpg'

windowSize = 100
step = 25

[patches, rectanglesX, rectanglesY] = splitImage(imPath, imName, windowSize, step)

rgb = patches[0]

hsv = matplotlib.colors.rgb_to_hsv(np.divide(rgb, 255))


print(np.max(rgb))

print(np.max(hsv))

