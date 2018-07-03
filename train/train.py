# -*- coding: utf-8 -*-

import numpy as np
import os
import keras
from keras import optimizers,initializers
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


Num_classes =4
img_nums = 137
img_w = 64
img_h =64
img_c = 1
epoch_num  =200
data_path = '../data_train/train.npy'
models_path = './models'
if os.path.exists(models_path) ==False:
    os.makedirs(models_path)


all_data = np.load(data_path)
x_train = all_data[:,:-1]
y_train = all_data[:,-1:]
y_train = to_categorical(y_train,num_classes = 4)


x_train = np.reshape(x_train,(img_nums,img_h,img_w,img_c))
x_test ,y_test = x_train,y_train

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(Num_classes))
model.add(Activation('softmax'))

# policy
adam =optimizers.Adam(lr=0.001)

model.compile(loss = 'categorical_crossentropy', optimizer = adam,metrics=['accuracy'])
# checkpoint
filepath=models_path+os.sep+"model-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
# fit
hist = model.fit(x_train, y_train,batch_size = 6 ,epochs=epoch_num,
                 shuffle=True,verbose=1,callbacks= callbacks_list)

file = open('./trian.log','w')
ctx = str(hist.history)
file.write(ctx)
file.close




