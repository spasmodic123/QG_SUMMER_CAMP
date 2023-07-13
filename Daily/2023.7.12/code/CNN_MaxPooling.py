import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten
from keras.src.layers import Conv2D, MaxPooling2D

model = Sequential([
    Dense(units=8,input_shape=(20,20,3),activation='relu'),
    Conv2D(16,kernel_size=(3,3),activation='relu',padding='same'),
    MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'),  # 2*2大小的滤波器,步长为2,一次滑动两个像素
    Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'),
    Flatten(),
    Dense(2,activation='softmax')
])
model.summary()
'''
结果
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 20, 20, 8)         32        
                                                                 
 conv2d (Conv2D)             (None, 20, 20, 16)        1168      
                                                                 
 max_pooling2d (MaxPooling2  (None, 10, 10, 16)        0         经过最大池操作,维度减少一半,维度到底减少多少滤波器size和步长stride
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 10, 10, 32)        4640      
                                                                 
 flatten (Flatten)           (None, 3200)              0         
                                                                 
 dense_1 (Dense)             (None, 2)                 6402  '''