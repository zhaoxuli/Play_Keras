# -*- coding: utf-8 -*-
import os
import cv2
import numpy   as np
out_path = '../data_train'
data_path = '../Images'
if not os.path.exists(out_path) :
    os.makedirs(out_path)
size =64
file_lst = os.listdir(data_path)
out_numpy = np.zeros(size*size+1)

count = 0
for i in  range (len(file_lst)):
    folder_path = data_path+os.sep+file_lst[i]
    img_lst = os.listdir(folder_path)
    print folder_path,'doing....'
    for img in img_lst:
        img_url = folder_path+os.sep+img
        src = cv2.imread(img_url,0)
        dst = cv2.resize(src,(size,size))
        dst = np.reshape(dst,(size*size,))
        dst = np.r_[dst,i]
        out_numpy = np.row_stack((out_numpy,dst))
        count +=1

out_numpy = out_numpy[1:]
np.save(out_path+os.sep+'train.npy',out_numpy)
print 'shape is ',
print out_numpy.shape,'img_num :',count



