# -*- coding: utf-8 -*-

from keras.models import load_model
import  numpy as np
import os
import cv2
#load model
model = load_model('./model-09-0.87.hdf5')
#get image to numpy
#        others is 0
#        Ash is    1
#        Cotton is 2
#        Grain is  3
label_lst = ['Others','Ash','Cotton','Grain']
img_size = [64,64,1]
folder_path = './Images'
imgs = os.listdir(folder_path)
data_in = np.zeros((img_size[0]*img_size[1]*img_size[2]))
print 'Your input image is:'
for ele in imgs:
    print ele[:-4],
    img_url = folder_path+os.sep+ele
    img = cv2.imread(img_url,0)
    img = cv2.resize(img,(img_size[0],img_size[1]))
    img = np.reshape(img,(1,img_size[0]*img_size[1]*img_size[2]))
    data_in = np.row_stack((data_in,img))
print ''
data_in = data_in[1:]
data_predict = np.reshape(data_in,(len(imgs),img_size[0],img_size[1],img_size[2]))
print 'The model predict result is:'
out = model.predict(data_predict)
out_lst= []
for ele in out:
    ele = list(ele)
    out_lst.append({'class':label_lst[ele.index(max(ele))],
                    'score':max(ele)})

for ele in out_lst:
    print ele['class'],('%.4f' % ele['score'])




