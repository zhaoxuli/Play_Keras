# -*- coding: utf-8 -*-

from keras.models import load_model
import  numpy as np
import os
import cv2
#load model
no_show =False
#model_path ='../../train/picked_models/model-145-0.95.hdf5'
model_path ='./model-09-0.87.hdf5'
models_floder = '../../train/picked_models'
#get image to numpy
#        others is 0
#        Ash is    1
#        Cotton is 2
#        Grain is  3
label_lst = ['Others','Ash','Cotton','Grain']
img_size = [64,64,1]
data_path = '../../test_images'

def show(model_path,data_predict):
    model = load_model(model_path)
    out = model.predict(data_predict)
    print 'The model predict result is:'
    nums = len(img_key_lst)
    out_lst= []
    #label_lst = ['Others','Ash','Cotton','Grain']
    info_lst = [{'label':0,'predict':0},{'label':0,'predict':0},{'label':0,'predict':0},{'label':0,'predict':0}]
    right_num = 0
    for i in range(nums):
        ele = list(out[i])
        info_lst[label_lst.index(img_key_lst[i].split(os.sep)[0])]['label'] +=1
        out_lst.append({'predict':label_lst[ele.index(max(ele))],
                        'label': img_key_lst[i],
                        'score':max(ele)})
        if  out_lst[i]['label'].split(os.sep)[0] == out_lst[i]['predict']:
            right_num += 1
            info_lst[label_lst.index(img_key_lst[i].split(os.sep)[0])]['predict'] +=1
            print ("%15s %15s %15s"  % (out_lst[i]['label'],out_lst[i]['predict'],out_lst[i]['score']))
    print '--------------------------------------------------------------------------------------------'
    for i in range(nums):
        if  out_lst[i]['label'].split(os.sep)[0] != out_lst[i]['predict']:
            print ("%15s %15s %15s"  % (out_lst[i]['label'],out_lst[i]['predict'],out_lst[i]['score']))
    print '--------------------------------------------------------------------------------------------'
    for i in range(4):
        print ("%15s %15s %15s %5s %15s" %(label_lst[i]+'_nums:',info_lst[i]['label'],'True_positive:',info_lst[i]['predict'],
                                            float(info_lst[i]['predict'])/float(info_lst[i]['label'])))
    print 'Acc is:',float(right_num)/float(len(img_key_lst))


def get_score(model_path,data_predict):
    model = load_model(model_path)
    out = model.predict(data_predict)
    print 'The model is:',model_path,
    nums = len(img_key_lst)
    out_lst= []
    #label_lst = ['Others','Ash','Cotton','Grain']
    info_lst = [{'label':0,'predict':0},{'label':0,'predict':0},{'label':0,'predict':0},{'label':0,'predict':0}]
    right_num = 0
    for i in range(nums):
        ele = list(out[i])
        info_lst[label_lst.index(img_key_lst[i].split(os.sep)[0])]['label'] +=1
        out_lst.append({'predict':label_lst[ele.index(max(ele))],
                        'label': img_key_lst[i],
                        'score':max(ele)})
        if  out_lst[i]['label'].split(os.sep)[0] == out_lst[i]['predict']:
            right_num += 1
            info_lst[label_lst.index(img_key_lst[i].split(os.sep)[0])]['predict'] +=1
    print float(right_num)/float(len(img_key_lst))
    return float(right_num)/float(len(img_key_lst))

if __name__ == '__main__':
    data_in = np.zeros((img_size[0]*img_size[1]*img_size[2]))
    folder_lst = os.listdir(data_path)
    img_key_lst = []
    for folder in folder_lst:
        folder_path = data_path+os.sep+folder
        imgs = os.listdir(folder_path)
        for ele in imgs:
            img_key_lst.append(folder+os.sep+ele)
            img_url = folder_path+os.sep+ele
            img = cv2.imread(img_url,0)
            img = cv2.resize(img,(img_size[0],img_size[1]))
            img = np.reshape(img,(1,img_size[0]*img_size[1]*img_size[2]))
            data_in = np.row_stack((data_in,img))
    data_in = data_in[1:]
    data_predict = np.reshape(data_in,(len(img_key_lst),img_size[0],img_size[1],img_size[2]))
    data_predict = data_predict
    if no_show:
        models = os.listdir(models_floder)
        acc_lst = []
        for ele in models:
            acc = get_score(models_floder+os.sep+ele,data_predict)
            acc_lst.append(acc)
        index = acc_lst.index(max(acc_lst))
        print 'the best model is:',models[index],'Acc:',max(acc_lst)
    else:
        show(model_path,data_predict)


