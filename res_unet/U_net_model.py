# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:44:45 2018

@author: czy
"""
# Import
from keras.models import Model
from keras.layers import Input, Conv2D,MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, add
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.losses import binary_crossentropy


from keras import backend as K
import tensorflow as tf
import yaml

with open('cfgs/cfg.yml', 'r') as f:
    unet_cfg = yaml.load(f)
### Resized Image dimension
image_rows = unet_cfg['image_rows']
image_cols = unet_cfg['image_cols']
image_channels = unet_cfg['image_channels']
### ----------------------------------------------------Build U-Net model----------------------------------------------------------
###
####### -----Loss Function----
# Dice Coefficient
  


def dice_coeff_0(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_coeff(y_true, y_pred): ###IOU
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth - intersection)
    return score

# Dice Coef Loss
def dice_coeff_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


# Loss function Combines Binary Crossentropy WITH Dice Loss
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_coeff_loss(y_true, y_pred)
    return loss

# Weighted Dice coefficient
def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score

# Weighted Dice Loss
def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss

# Weighted Binary Crossentropy  Loss
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


# Weighted Binary Crossentropy  Loss  (my own program to hardcode weight in the weighted bce loss)
def weighted_bce_loss_ver2(y_true, y_pred):

    # weight computed by Matlab
    weight =  19.0
    
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))


    ### tensor flow: weight bce loss
    # let x = logits, z = targets, q = pos_weight.
    # let l = (1 + (q - 1) * z)
    # The loss is:
    # (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    
    return K.sum(loss) / K.sum(weight)




# Weighted Binary Crossentropy Loss (with weighted boundary) Combined WITH Dice Loss
def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss








### -------Unet Model Layer model-----
def UnetResidual_model(input_shape=(image_rows,image_cols,image_channels),num_classes=1,  k=64,kc=512):

    # Conv filter parameters
    k1=k
    k2=2*k1
    k3=2*k2
    k4=2*k3
    kc=2*k4
    #print('k1 ={}'.format(k1))
    #print('k2 ={}'.format(k2))
    #print('k3 ={}'.format(k3))
    #print('k4 ={}'.format(k4))
    #print('kc ={}'.format(kc))


    
    #### Build U-Net model
    #### Input
    inputs = Input((image_rows,image_cols , image_channels)) 

    #### DownSampling Block (4) 
    down1 = Conv2D(k1, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(k1, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    down2 = Conv2D(k2, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(k2, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    shortcut = Conv2D(k2, (1, 1))(down1_pool)
    shortcut = BatchNormalization()(shortcut)
    down2 = add([down2, shortcut])                                  # skip connection
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(k3, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(k3, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    shortcut = Conv2D(k3, (1, 1))(down2_pool)
    shortcut = BatchNormalization()(shortcut)    
    down3 = add([down3, shortcut])                                  # skip connection
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    down4 = Conv2D(k4, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(k4, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    shortcut = Conv2D(k4, (1, 1))(down3_pool)
    shortcut = BatchNormalization()(shortcut)        
    down4 = add([down4, shortcut])                                  # skip connection
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    down4_pool = Dropout(0.5)(down4_pool)

    #### Center Block
    center = Conv2D(kc, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(kc, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    shortcut = Conv2D(kc, (1, 1))(down4_pool)
    shortcut = BatchNormalization()(shortcut)            
    center = add([center, shortcut])                                  # skip connection
    center = Activation('relu')(center)
    center = Dropout(0.5)(center)
    

    #### UpSampling Blocks (4)
    up4 = UpSampling2D((2, 2))(center)
    up4_con = concatenate([down4, up4], axis=3)
    up4 = Conv2D(k4, (3, 3), padding='same')(up4_con)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(k4, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(k4, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    shortcut = Conv2D(k4, (1, 1))(up4_con)
    shortcut = BatchNormalization()(shortcut)                
    up4 = add([up4, shortcut])                                         # skip connection   
    up4 = Activation('relu')(up4)

    up3 = UpSampling2D((2, 2))(up4)
    up3_con = concatenate([down3, up3], axis=3)
    up3 = Conv2D(k3, (3, 3), padding='same')(up3_con)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(k3, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(k3, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    shortcut = Conv2D(k3, (1, 1))(up3_con)
    shortcut = BatchNormalization()(shortcut)                    
    up3 = add([up3, shortcut])                                         # skip connection 
    up3 = Activation('relu')(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2_con = concatenate([down2, up2], axis=3)
    up2 = Conv2D(k2, (3, 3), padding='same')(up2_con)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(k2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(k2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    shortcut = Conv2D(k2, (1, 1))(up2_con)
    shortcut = BatchNormalization()(shortcut)                        
    up2 = add([up2, shortcut])                                          # skip connection
    up2 = Activation('relu')(up2)

    up1 = UpSampling2D((2, 2))(up2)
    up1_con = concatenate([down1, up1], axis=3)
    up1 = Conv2D(k1, (3, 3), padding='same')(up1_con)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(k1, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(k1, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    shortcut = Conv2D(k1, (1, 1))(up1_con)
    shortcut = BatchNormalization()(shortcut)                            
    up1 = add([up1, shortcut])                                           # skip connection
    up1 = Activation('relu')(up1)
    
    # Pixel Classification 
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    # Model
    model = Model(inputs=[inputs], outputs=[classify])         # optimizer=sgd

    ### Optimizer, Loss and Metrics
    model.compile(optimizer=Adam(lr=1e-3), loss= binary_crossentropy, metrics=[dice_coeff])

    # Display the model
    model.summary()
    
    return model
