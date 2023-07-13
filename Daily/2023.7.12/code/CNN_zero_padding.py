# 零填充  valid_padding和same_padding,   valid代表不填充,输出维度减少(默认default)   same代表零填充,输出维度保持不变
import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten
from keras.src.layers import Conv2D

#valid模式
model_valid = Sequential([
    Dense(units=16,input_shape=(20,20,3),activation='relu'),  # Dense是一种密集层
    Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'),  # Conv2D是一种卷积层,kernel_size代表滤波器的size,padding代表零填充的模式
    Conv2D(64,kernel_size=(5,5),padding='valid',activation='relu'),
    Conv2D(128,kernel_size=(7,7),padding='valid',activation='relu'),
    Flatten(),
    Dense(2,activation='softmax')]
)
model_valid.summary()
'''结果
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 20, 20, 16)        64        刚开始的尺寸(维度)是20*20,使用的是valid填充模式
                                                                 
 conv2d (Conv2D)             (None, 18, 18, 32)        4640      讲过3*3滤波器,维度下降到18*18
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 14, 64)        51264     维度下降到14*14
                                                                 
 conv2d_2 (Conv2D)           (None, 8, 8, 128)         401536    维度下降到8*8
                                                                 
 flatten (Flatten)           (None, 8192)              0         
                                                                 
 dense_1 (Dense)             (None, 2)                 16386  '''


# same模式
model_valid = Sequential([
    Dense(units=16,input_shape=(20,20,3),activation='relu'),  # Dense是一种密集层
    Conv2D(32,kernel_size=(3,3),padding='same',activation='relu'),  # Conv2D是一种卷积层,kernel_size代表滤波器的size,padding代表零填充的模式
    Conv2D(64,kernel_size=(5,5),padding='same',activation='relu'),
    Conv2D(128,kernel_size=(7,7),padding='same',activation='relu'),
    Flatten(),
    Dense(2,activation='softmax')]
)
model_valid.summary()
'''
结果
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_2 (Dense)             (None, 20, 20, 16)        64        图片维度保持在20*20,不改变
                                                                 
 conv2d_3 (Conv2D)           (None, 20, 20, 32)        4640      
                                                                 
 conv2d_4 (Conv2D)           (None, 20, 20, 64)        51264     
                                                                 
 conv2d_5 (Conv2D)           (None, 20, 20, 128)       401536    
                                                                 
 flatten_1 (Flatten)         (None, 51200)             0         
                                                                 
 dense_3 (Dense)             (None, 2)                 102402   
'''