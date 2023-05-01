#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
x = np.random.rand(30)
y = np.random.rand(30)
plt.scatter(x, y, s=300, alpha=0.5, linewidths=2, c='#aaaaFF', edgecolors='b')


# In[ ]:





# In[12]:


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os#.os模組
import random
import numpy as np
import keras
from PIL import Image#PIL提供處理image的模組
from keras.models import Sequential
from keras.layers  import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D#.CNN·的捲基層和池化層
from keras.models import load_model#載入模型
from keras.utils import np_utils#來後續將Label.標簽轉為.one-hot-
from matplotlib import pyplot as plt
#.data_x舆data_y(Label)前處理函式
def data_x_y_preprocess (datapath):
    img_row, img_col = 28, 28
    datapath = datapath
    data_x = np.zeros((28,28)).reshape(1,28,28)
    pictureCount = 0
    data_y = []
    num_class = 10
    for root, dirs, files in os.walk(datapath):
        for f in files :
            label = int(root.split("\\")[-1])
            data_y.append(label)
            fullpath=os.path.join(root,f)
            img = Image.open(fullpath)
            img=(np.array(img)/255).reshape(1,28,28)
            data_x=np.vstack((data_x,img))
            pictureCount += 1
    data_x=np.delete(data_x, [0],0)
    data_x=data_x.reshape(pictureCount, img_row, img_col, 1)
    data_y=np_utils.to_categorical(data_y,num_class)
    return data_x, data_y
#.建立簡單的線性執行的模型
model = Sequential()
#.建立卷積層,filter=32,即.outputspace深度,KernalSize3x3,activationfunction
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
#.建立池化層,池化大小=2x2,取最大值
model.add(MaxPooling2D(pool_size=(2,2)))
#建立卷積層,filter=64,即outputsize,KernaSize3x3,activationfuncton採
model.add(Conv2D(64,(3,3), activation='relu'))
#.建立池化層,池化大小=2x2,取最大值
model.add(MaxPooling2D(pool_size=(2,2)))
#Dropout層随機斷開輸入神經元,用於防止過度擬合,斷開比例:0.25
model.add(Dropout(0.1))
#Flatten層把多維的輸入一維化,常用在從卷積層到全連接層的過渡。
model.add(Flatten())
#·Dropout層隨機斷開輸人神經元,用於防止過度擬合,斷開比例:0.1
model.add(Dropout(0.1))
#.全連接層:128個output
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
#・使用·softmax activation.function,將結果分類(units=10.表示分10類)
model.add(Dense(units=10,activation='softmax'))
data_train_X, data_train_Y = data_x_y_preprocess("\\Users\\Ericw\\Dropbox\\Desktop-win10\\handwrite__detect\\train_image")

data_test_X, data_test_Y = data_x_y_preprocess("\\Users\\Ericw\\Dropbox\\Desktop-win10\\handwrite__detect\\test_image")
model.compile(loss="categorical_crossentropy",optimizer="adam" ,metrics=[ 'accuracy'])
train_history = model.fit(data_train_X,data_train_Y,batch_size=32,epochs=150,verbose=1,validation_split=0.1)
score = model.evaluate(data_test_X,data_test_Y,verbose=0)
print('Test loss:' ,score[0])
print('Test accuracy: ', score[1])
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt. legend(['loss', 'val_loss'], loc='upper left')
plt.show()


# In[ ]:




