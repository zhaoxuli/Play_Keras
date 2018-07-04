# -*- coding: utf-8 -*-

import os

path = './models'
out_folder = './picked_models'
if os.path.exists(out_folder) ==False:
    os.makedirs(out_folder)
models = os.listdir(path)

def cmp(a,b):
    ao = int(a.split('.')[1])
    bo = int(b.split('.')[1])
    return 1 if ao> bo else -1

models.sort(cmp,reverse=True)

for ele in  models[0:10]:
    cmd = 'mv '+path+os.sep+ele+' '+out_folder
    print cmd
    os.system(cmd)
