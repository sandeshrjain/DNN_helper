# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:06:30 2022

@author: Sandesh Jain
"""



from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Cropping2D,
                                     Lambda, 
                                    MaxPooling2D, Dropout, Concatenate)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.image import resize

train_gen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)
test_gen = ImageDataGenerator(rescale = 1./255)

train_set = train_gen.flow_from_directory('./sports-video-data/train_images',
                                            target_size=(64, 64),
                                            batch_size= 32)

test_set = test_gen.flow_from_directory('./sports-video-data/test_images',
                                            target_size=(64, 64),
                                            batch_size= 32)
def resize_layer(tensor):
    print('before shape', tensor.shape)
    tensor = resize(tensor, (64,64))
    print('after shape', tensor.shape)
    return tensor 


def trainer(train_set=train_set, test_set=test_set, shape=(64,64, 3), labels=5, model_name= 'Default',  channels=[32,64,128], 
            kernel_shapes = [(3,3), (3,3),(3,3)],
            pools = [1, 1, 1],
            activations=['relu','relu', 'relu', 'softmax'], drops=[0.1, 0.1, 0.1 ,0.1], 
            epochs=40, batch_size=32, optim='Adam', 
            loss='categorical_crossentropy', shear_range = 0.2, zoom_range = 0.2):
  print('Model Config: ', model_name)
#  assert(len(activations) == len(drops) == len(pools) + 1 == len(channels) + 1 == len(kernel_shapes) +1)

  model_input = Input(shape)
  u = model_input
  d = model_input
  d = Cropping2D(cropping=((16,16),(16,16)))(d)
  d = Lambda(resize_layer, name="lambda_layer")(d)
  for channel, kernel_shape, activation, drop, pool in zip(channels, kernel_shapes, activations, drops, pools):
      u = Conv2D(channel, kernel_shape, strides=(1,1), activation = activation, padding = 'same')(u)
      u = Dropout(drop)(u)
      d = Conv2D(channel, kernel_shape, strides=(1,1), activation = activation, padding = 'same')(d)
      d = Dropout(drop)(d)
      if pool: 
          u = MaxPooling2D()(u)
          d = MaxPooling2D()(d)
  c = Concatenate(axis= -1)([u, d])
  c = Flatten()(c)
  c = Dropout(drops[-1])(c)
  c = Dense(labels, activation = activations[-1])(c)
  
  model = Model(model_input, c)

  model.compile(optimizer= optim, loss=loss, metrics=['accuracy'])
  history = model.fit_generator(train_set,
                        epochs = epochs,
                        validation_data =test_set)
  
  
  model.save('sport_multi_saved_model/' + model_name)
  
  # convert the history.history dict to a pandas DataFrame:     
  hist_df = pd.DataFrame(history.history) 
  hist_csv_file = 'sport_multi_saved_model/'+model_name+'.csv'
  with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model_dict = {'Main': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
              
            'Channels_x2': {'channels':[64,128,256], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Channels_x4': {'channels':[128,256,512], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Channels_x8': {'channels':[256,512,1024], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Channels_custom': {'channels':[128,128,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Conv_Layers_add_1': {'channels':[32,64,128, 256], 
            'kernel_shapes': [(3,3), (3,3),(3,3), (3,3)],
            'pools': [1, 1, 1, 0],
            'activations': ['relu','relu', 'relu', 'relu', 'softmax'], 
            'drops': [0.1, 0.1, 0.1 ,0.1, 0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Conv_Layers_add_2': {'channels':[32,64,128, 256, 512], 
            'kernel_shapes': [(3,3), (3,3),(3,3), (3,3), (3,3)],
            'pools': [1, 1, 1, 1, 0],
            'activations': ['relu','relu', 'relu', 'relu', 'relu', 'softmax'], 
            'drops': [0.1, 0.1, 0.1 ,0.1, 0.1, 0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_4': {'channels':[32,64,128], 
            'kernel_shapes': [(4,4), (4,4),(4,4)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_5': {'channels':[32,64,128], 
            'kernel_shapes': [(5,5), (5,5),(5,5)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_6': {'channels':[32,64,128], 
            'kernel_shapes': [(6,6), (6,6),(6,6)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_7': {'channels':[32,64,128], 
            'kernel_shapes': [(7,7), (7,7),(7,7)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Kernel_custom_1': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3),(5,5),(7,7)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Drop_0_2': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.2, 0.2, 0.2 ,0.2], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Drop_0_3': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.3, 0.3, 0.3 ,0.3], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Drop_0_4': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.4, 0.4, 0.4 ,0.4], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Drop_0_5': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.5, 0.5, 0.5 ,0.5], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adam', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_SGD': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'SGD', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_RMSProp': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'RMSprop', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_Adadelta': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adadelta', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            },
            
            'Optim_Adagrad': {'channels':[32,64,128], 
            'kernel_shapes': [(3,3), (3,3),(3,3)],
            'pools': [1, 1, 1],
            'activations': ['relu','relu', 'relu', 'softmax'], 'drops': [0.1, 0.1, 0.1 ,0.1], 
            'epochs': 60, 'batch_size': 32, 'optim': 'Adagrad', 
            'loss': 'categorical_crossentropy', 'shear_range': 0.2, 'zoom_range': 0.2
            }}



model_dict_named = {'Channels Experiment':    ('channels', ['Main', 'Channels_x2', 'Channels_x4', 'Channels_x8', 'Channels_custom']),
                    'Layers Experiment':      ('channels', ['Main', 'Conv_Layers_add_1', 'Conv_Layers_add_2']),
                    'Dropout Experiment':     ('drops', ['Main', 'Drop_0_2','Drop_0_3', 'Drop_0_4','Drop_0_5']),
                    'Kernel Shape Experiment':('kernel_shapes', ['Main', 'Kernel_4', 'Kernel_5','Kernel_6', 'Kernel_7', 'Kernel_custom_1']),
                    'Optimizer Experiment':   ( 'optim', ['Main', 'Optim_SGD', 'Optim_RMSProp', 'Optim_Adadelta', 'Adagrad'])}


for model_name in model_dict:
    trainer(model_name = model_name, channels= model_dict[model_name]['channels'], 
            kernel_shapes=model_dict[model_name]['kernel_shapes'],
            pools=model_dict[model_name]['pools'],
            activations= model_dict[model_name]['activations'],
            drops=model_dict[model_name]['drops'],
            epochs= 1, optim=model_dict[model_name]['optim'])
    

            
             