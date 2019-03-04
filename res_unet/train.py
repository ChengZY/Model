# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:01:09 2018

@author: czy
"""
import os
import sys
import random
import warnings

import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imsave, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
# Import
from keras.models import Model
from keras.layers import Input, Conv2D,Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, add
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.losses import binary_crossentropy

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
import tensorflow as tf

import Image_preprocess
import U_net_model

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Unet network')
  parser.add_argument('--train', dest='train',
                      help='train dataset',
                      default='../data/train', type=str)
  parser.add_argument('--val', dest='val',
                      help='validation dataset',
                      default='../data/valid', type=str)
  parser.add_argument('--output', dest='output',
                      help='output model',
                      default='./Models/', type=str)
  parser.add_argument('--epoch', dest='epoch',
                      help='num of epoch to train',
                      default=200, type= int)
  parser.add_argument('--bs', dest='bs',
                      help='batch size',
                      default=2, type= int)
  parser.add_argument('--cfg', dest='cfg',
                      help='config',
                      default='cfg', type=str)

  args = parser.parse_args()
  return args

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = yaml.load(f)
  return yaml_cfg
    
args = parse_args()
args.cfg_file = "cfgs/{}.yml".format(args.cfg)
if args.cfg_file is not None:
    unet_cfg = cfg_from_file(args.cfg_file)

####Init parameter ###
K.set_image_data_format('channels_last')     # TF dimension ordering in this code
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
SEED = unet_cfg['SEED']
Raw_channels= unet_cfg['Raw_channels']
image_rows = unet_cfg['image_rows']
image_cols = unet_cfg['image_cols']
image_channels = unet_cfg['image_channels']
k = unet_cfg['k']
num_classes=unet_cfg['num_classes']
threshold= unet_cfg['threshold']
epochs = args.epoch
batch_size = args.bs
ratio = unet_cfg['ratio']

#### Define a combined data generator
#### function to combine generators (https://github.com/keras-team/keras/issues/5720)
def combine_generator(gen1, gen2):
    while True:                                                       ### BE SURE TO add 'True' to make it a loop 
        yield(gen1.next(), gen2.next())     ### yield will make this function a generator and iteratively produce next value without end, NOT the same batch.
###Load and Prepocess Images

train_imgs_path   =     os.path.join(args.train,'Images/')
train_masks_path  =     os.path.join(args.train,'Masks')
#pred_path        =     './PredMasks/2'
val_imgs_path     =     os.path.join(args.val,'Images/')
val_masks_path    =     os.path.join(args.val,'Masks')

train_imgs_array, train_masks_array, sizes_val = Image_preprocess.load_img_masks(train_imgs_path, train_masks_path, folder = str(1))
mean = np.mean(train_imgs_array.astype('float32'))
std  = np.std(train_imgs_array.astype('float32'))

if os.path.exists(args.val):
    val_imgs_array, val_masks_array, sizes_val = Image_preprocess.load_img_masks(val_imgs_path, val_masks_path, folder = str(1) )

else: 
    
    val_imgs_array = train_imgs_array[int(ratio*len(train_imgs_array)):]
    val_masks_array = train_masks_array[int(ratio*len(train_masks_array)):]
    
    train_imgs_array = train_imgs_array[:int(ratio*len(train_imgs_array))]
    train_masks_array = train_masks_array[:int(ratio*len(train_masks_array))]

### ---------------------------------Model Training (Data Augmentation)-----------------------------
### Data Augmentation (https://keras.io/preprocessing/image/)
# we create two instances with the same arguments
#### load U-net model
model = U_net_model.UnetResidual_model(input_shape=(image_rows,image_cols,image_channels),num_classes=1,  k=k)
#### checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=150)
model_checkpoint = ModelCheckpoint(os.path.join(args.output,'unetRes_plate_{}x{}_k{}.h5'.format(image_rows,image_cols,k)),\
                                   monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1)  
if os.path.exists(os.path.join(args.output,'unetRes_plate_{}x{}_k{}.h5'.format(image_rows,image_cols,k))):
    model.load_weights(os.path.join(args.output,'unetRes_plate_{}x{}_k{}.h5'.format(image_rows,image_cols,k)))
######training on the augmented data
model.fit_generator(Image_preprocess.train_generator(train_imgs_array,train_masks_array,batch_size),
                    steps_per_epoch=len(train_imgs_array) // batch_size,
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    validation_data = Image_preprocess.val_generator(val_imgs_array,val_masks_array,batch_size),
                    validation_steps= val_imgs_array.shape[0] // batch_size,
                    callbacks=[early_stopping,model_checkpoint])
