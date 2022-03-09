# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 22:45:55 2022

@author: Sandesh Jain
"""

from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input,
                                     Lambda, MaxPooling2D, Dropout, Concatenate)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


cif10 = cifar10.load_data() 

x_train10 = cif10[0][0]
y_train10 = cif10[0][1]
x_test10 = cif10[1][0]
y_test10 = cif10[1][1]

#x_train10[1].shape
#change label encoding

y_train10 = to_categorical(y_train10)

y_test10 = to_categorical(y_test10)
y_train10.shape

train_gen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)

test_gen = ImageDataGenerator(rescale = 1./255)

train_set = train_gen.flow(x_train10, y_train10,
                                            batch_size= 32)

test_set = test_gen.flow(x_test10, y_test10,
                                            batch_size= 32)

def trainer(X=x_train10, y=y_train10, x_val=x_test10, 
            y_val=y_test10, model_name= 'Default',  channels=[32,64,128], 
            kernel_shapes = [(3,3), (3,3),(3,3)],
            pools = [1, 1, 1],
            activations=['relu','relu', 'relu', 'softmax'], drops=[0.1, 0.1, 0.1 ,0.1], 
            epochs=40, batch_size=64, optim='Adam', 
            loss='categorical_crossentropy', shear_range = 0.2, zoom_range = 0.2):
  print('Model Config: ', model_name)
  assert(len(activations) == len(drops) == len(pools) + 1 == len(channels) + 1 == len(kernel_shapes) +1)
  train_gen = ImageDataGenerator(rescale =1./255,
                                   shear_range =shear_range,
                                   zoom_range = zoom_range,
                                   horizontal_flip =True)

  test_gen = ImageDataGenerator(rescale = 1./255)

  train_set = train_gen.flow(x_train10, y_train10,
                                            batch_size= batch_size)

  test_set = test_gen.flow(x_test10, y_test10,
                                            batch_size= batch_size)
  model_input = Input(np.shape(X[0]))
  u = model_input
  d = model_input
  for channel, kernel_shape, activation, drop, pool in zip(channels, kernel_shapes, activations, drops, pools):
      u = Conv2D(channel, kernel_shape, strides=(1,1), activation = activation, padding = 'same')(u)
      d = Conv2D(channel, kernel_shape, strides=(1,1), activation = activation, padding = 'same')(d)
      u = Dropout(drop)(u)
      d = Dropout(drop)(d)
      if pool: 
          u = MaxPooling2D()(u)
          d = MaxPooling2D()(d)
  c = Concatenate(axis= -1)([u, d])
  c = Flatten()(c)
  c = Dropout(drops[-1])(c)
  c = Dense(np.shape(y)[1], activation = activations[-1])(c)
  
  model = Model(model_input, c)
  model.compile(optimizer= optim, loss=loss, metrics=['accuracy'])
  history = model.fit_generator(train_set,
                        epochs = epochs,
                        validation_data =test_set)
  
  
  model.save('saved_model/' + model_name)
  
  # convert the history.history dict to a pandas DataFrame:     
  hist_df = pd.DataFrame(history.history) 
  hist_csv_file = model_name+'.csv'
  with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model_dict = {'Main': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
              
            'Channels_x2': {'channels':[64,128,256], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Channels_x4': {'channels':[128,256,512], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Conv_Layers_add_1': {'channels':[32,64,128, 256], 
            'kernel_shapes': [(3,3), (3,3),(3,3), (3,3)],
            'pools': [1, 1, 0, 0],
            'activations': ['relu','relu', 'relu', 'relu', 'softmax'], 
            'drops': [0.1, 0.1, 0.1 ,0.1, 0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Conv_Layers_add_2': {'channels':[32,64,128, 256, 512], 
            'kernel_shapes': [(3,3), (3,3),(3,3), (3,3), (3,3)],
            'pools': [1, 1, 0, 0, 0],
            'activations': ['relu','relu', 'relu', 'relu', 'relu', 'softmax'], 
            'drops': [0.1, 0.1, 0.1 ,0.1, 0.1, 0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_5': {'channels':[32,64,128], 
            'kernel_shapes': [(5,5), (5,5),(5,5)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_7': {'channels':[32,64,128], 
            'kernel_shapes': [(7,7), (7,7),(7,7)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Drop_0_2': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.2, 0.2, 0.2 ,0.2], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Drop_0_4': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.4, 0.4, 0.4 ,0.4], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_SGD': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'SGD', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_RMSProp': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'RMSprop', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_Adadelta': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 64, 'optim': 'Adadelta', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            }}



model_dict_named = {'Channels Experiment':    ('channels', ['Main', 'Channels_x2', 'Channels_x4']),
                    'Layers Experiment':      ('channels', ['Main', 'Conv_Layers_add_1', 'Conv_Layers_add_2']),
                    'Dropout Experiment':     ('drops', ['Main', 'Drop_0_2', 'Drop_0_4']),
                    'Kernel Shape Experiment':('kernel_shapes', ['Main', 'Kernel_5', 'Kernel_7']),
                    'Optimizer Experiment':   ( 'optim', ['Main', 'Optim_SGD', 'Optim_RMSProp', 'Optim_Adadelta'])}


for model_name in model_dict:
    trainer(model_name = model_name, channels= model_dict[model_name]['channels'], 
            kernel_shapes=model_dict[model_name]['kernel_shapes'],
            pools=model_dict[model_name]['pools'],
            activations= model_dict[model_name]['activations'],
            drops=model_dict[model_name]['drops'],
            epochs= 1, optim=model_dict[model_name]['optim'])
    


def Experiment_Visualize(model_dict, model_dict_named):
  for exp in model_dict_named:
    legend_list = []
    model_set = model_dict_named[exp][1]
    exp_entity = model_dict_named[exp][0]
    _, plt_train = plt.subplots()
    _, plt_test = plt.subplots()
    for m in model_set:
      data = pd.read_csv('/saved_model/' + m + '.csv')
      legend_list.append(model_dict[m][exp_entity])
      # summarize history for accuracy

      plt_train.plot(data['accuracy'])
      plt_test.plot(data['val_accuracy'])
    plt_train.set_title('Training Accuracy: ' + exp)
    plt_test.set_title('Testing Accuracy: ' + exp)

    plt_train.set_ylabel('accuracy')
    plt_test.set_ylabel('accuracy')
  
    plt_train.set_xlabel('epoch')
    plt_test.set_xlabel('epoch')

    plt_train.legend(legend_list, loc='center left', bbox_to_anchor=(1, 0.5))
    plt_test.legend(legend_list, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

