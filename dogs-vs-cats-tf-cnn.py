# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:08:04 2022

@author: Sandesh Jain
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input,
                                     Lambda, MaxPooling2D, Dropout)

def make_data():
    path, dirs, files = next(os.walk("C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train/train"))
    #file_count = len(files)
    for file in files:
        if file[0] == 'c':
            os.rename("C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train/train/" + file,
                      "C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train/cat/" + file)
        else:
            os.rename("C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train/train/" + file,
                      "C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train/dog/" + file)

    for label in ['dog', 'cat']:
        path, dirs, files = next(os.walk("C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train/" + label))
        #file_count = len(files)
        for idx, file in enumerate(files):
            if idx >= 10000:
                os.rename(path + '/' + file,
                          "C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/test/" + label + '/' + file)



train_gen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)

test_gen = ImageDataGenerator(rescale = 1./255)

train_set = train_gen.flow_from_directory('C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/train',
                                            target_size=(64,64),
                                            batch_size= 32,
                                            class_mode='binary')

test_set = test_gen.flow_from_directory('C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM3/dump/dogs-vs-cats/test',
                                            target_size=(64,64),
                                            batch_size= 32,
                                            class_mode='binary')


def build_cat_dog_model():
    input_img = Input((64,64,3))
    x = Conv2D(32, (5, 5), strides=(1,1), padding = 'valid', activation='relu')(input_img)
    x = Conv2D(32, (5, 5), strides=(1,1), padding = 'valid', activation='relu')(x)

    x = MaxPooling2D()(x)
    
    x = Conv2D(64, (3,3), strides=(1,1), padding = 'valid', activation='relu')(x)
    x = Conv2D(64, (3,3), strides=(1,1), padding = 'valid', activation='relu')(x)
    
    x = MaxPooling2D()(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(input_img, x)
    
    model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model 

dc_model = build_cat_dog_model()

dc_model.fit_generator(train_set,
                        epochs = 30,
                        validation_data =test_set)
    
    






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    