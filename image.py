# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:42:40 2018

@author: admin
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('plastic.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'Blues')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
