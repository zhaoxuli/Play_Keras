# -*- coding: utf-8 -*-
import cv2
import numpy as np

a = np.load('./train.npy')
a = a[:,:-1]
b = np.reshape(a,(137,64,64,1))
b = b[5,:,:].astype(np.uint8)
print b.shape
cv2.imshow('b',b)
cv2.waitKey()

