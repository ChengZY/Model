# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:57:42 2018

@author: czy
"""

import os
import warnings

import numpy as np
from skimage.transform import resize
from imgaug import augmenters as iaa

from keras.preprocessing import image
from keras import backend as K
import yaml


with open('cfgs/cfg.yml', 'r') as f:
    unet_cfg = yaml.load(f)


# Image Format 'channels_last'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
# Set Random Seed
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
SEED = unet_cfg['SEED']
### Raw Image Dimensions
### Raw image have different dimensions
Raw_channels= unet_cfg['Raw_channels']
### Resized Image dimension
image_rows = unet_cfg['image_rows']
image_cols = unet_cfg['image_cols']
image_channels = unet_cfg['image_channels']
# Binary Mask
num_classes=unet_cfg['num_classes']
mean = unet_cfg['mean'], 
std = unet_cfg['std']

# ---------------------------------------------------------------------------------------------

def read_single_images(imgs_path, filename, dot = '.jpg', grayscale= True, channel = 0):
    image_file_name = filename.split('.')[0] + dot  # '1.jpg'
    full_image_path = os.path.join(imgs_path, image_file_name)
    #    print(full_image_path)
    ### Raw-size image
    # img = imread(full_image_path, as_grey=True)
    img = image.load_img(full_image_path, target_size=None, grayscale=grayscale)    
    img = image.img_to_array(img)    
    img_array = np.zeros((img.shape[0], img.shape[1], image_channels), dtype=np.uint8)
    if not grayscale:
        img_array[:,:,0] = img[:,:,channel]
        img = img_array
    img = np.squeeze(img).astype(np.uint8)
    # Store Raw train image size
    sizes = [img.shape[0], img.shape[1]]
    # resize the image to the specified dimensions
    img = resize(img, (image_rows, image_cols), mode='constant', preserve_range=True)
    img = img[..., np.newaxis]
    img = np.array([img])  # Add a dimension at the front
    
    return img, sizes

def load_img(imgs_path):
### ------------------------------Load & Display training images and the corresponding training masks-----------------------
# list of file names including suffix
    files = os.listdir(imgs_path)
    total = len(files)    
    imgs_array = np.zeros((total, image_rows, image_cols, image_channels), dtype=np.uint8)
    sizes_test = []
    print('loading... testing dataset')
    i = 0
    for filename in files:
        print(filename)
        img, sizes = read_single_images(imgs_path, filename)
        sizes_test.append(sizes)
        imgs_array[i] = img
        if i % 20 == 0:
            print('Done: {}/{} images'.format(i, total))
        i = i + 1
        
    return imgs_array, sizes_test

def load_img_masks(imgs_path, masks_path, folder):
### ------------------------------Load & Display training images and the corresponding training masks-----------------------
# list of file names including suffix
    files = os.listdir(imgs_path)
    total = len(files)    
    imgs_array = np.zeros((total, image_rows, image_cols, image_channels), dtype=np.uint8)
    masks_array = np.zeros((total, image_rows, image_cols, image_channels), dtype=np.uint8)
    sizes_test = []
    print('loading... testing dataset')
    i = 0
    for filename in files:
        img, sizes = read_single_images(imgs_path, filename)
        mask, sizes = read_single_images(os.path.join( masks_path, folder), filename, dot = '.jpg', grayscale= False)
        sizes_test.append(sizes)
        imgs_array[i] = img
        masks_array[i] = mask
        if i % 5 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i = i + 1
        
    return imgs_array, masks_array, sizes_test


##### ---------------------------------------------------------Image Processing------------------------------------------------------------
#imgs_array = load_img(test_imgs_path)
#### set mean and std
# ('mean= ' + str(mean))  #
# ('std = ' + str(std))   #
def Normalization_img(imgs_array, mean = mean, std = std):
    print('mean of data pass = ' + str(mean))  # 117
    print('std  of data pass = ' + str(std))   # 60       #### 'mean' and 'std' are used in
    imgs_array = imgs_array.astype('float32')
    ### Normalize
    imgs_array -= mean
    imgs_array /= std
    return imgs_array

### Scale masks to 0~1
def binary_mask(masks_array):
    masks_array = masks_array.astype('float32')
    masks_array /= 255.
    return masks_array


seq_img_only = iaa.OneOf(
       [
        ### 1. Add gaussian noise (aka white noise) to images.
        iaa.AdditiveGaussianNoise(scale=0.01*255),
        ### 2. Add random values between -40 and 40 to images
        iaa.AddElementwise((-20, 20)),
        ### 3. Multiply all pixels in an image with a specific value, thereby making the image darker or brighter 
        iaa.Multiply((0.5, 1.5)),
        ### 4. Multiply each pixel with a random value between 0.5 and 1.5
        iaa.MultiplyElementwise((0.5, 1.5)),       
        ### 4. Sharpen an image, then overlay the results with the original using an alpha between 0.0 and 1.0
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.8, 1.2)),
        ### 5. Augmenter that embosses images and overlays the result with the original image
        #iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
        ### 6. blur images using gaussiankernels
        iaa.GaussianBlur(sigma=(0.5, 1.5)),
        ### 7. blur images using median kernels
        #iaa.MedianBlur(k=(3, 5)),
        ### 8. blur images using average kernals
        iaa.AverageBlur(k=(2, 3)),
#        ### 9. ElasticTransformation by moving pixels locally around using displacement fields
#        iaa.ElasticTransformation(alpha=(0, 2.0), sigma=0.10),
        ### 10. changes the contrast of images
        iaa.ContrastNormalization((0.8, 1.2))        
       ])

#### Affine Transformation:
#### Applied to BOTH Imgs and Masks: Define a sequence of augmentations
#### (Involve crops, flips, and affine transformation)
seq_both = iaa.Sequential([
                           #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
                           iaa.Fliplr(0.5),      # horizontally flip 50% of the images
                           iaa.Affine(
                                      scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                      translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                      rotate=(-10, 10),
                                      shear=(-20, 20)
                                     )    
                         ])
                           
def train_generator(train_imgs_array,train_masks_array,batch_size):
    while True:
        ## Random seed for data generation
        seed_gen=np.random.randint(10000,size=1)[0]
        #print('[Data Generation] seed_gen = ' + str(seed_gen))
        ## random shuffle data
        np.random.seed(seed=seed_gen)
        np.random.shuffle(train_imgs_array)
        ## random shuffle labels
        np.random.seed(seed=seed_gen)
        np.random.shuffle(train_masks_array)
        
        ## This is the start of each batch
        for start in range(0, len(train_imgs_array), batch_size):
            ## Initialize a batch of images and masks
            ## The target is to generate a batch and then yield. 
            x_batch = []
            y_batch = []            
            ## This is the end of the batch
            end = min(start + batch_size, len(train_imgs_array))
            ## img & masks indices in the current batch
            ids_batch = [i for i in range(start,end)]
            ## Convert the stochastic sequence of augmenters to a deterministic one.
            ## The deterministic sequence will always apply the exactly same effects to the images.
            ## NOTE: call this for each batch again, NOT only once at the start
            seq_det_both = seq_both.to_deterministic() 
            ## Load the batch of images and masks
            for id in ids_batch:                                       
                ### img & mask
                img   = np.squeeze(train_imgs_array[id])
                mask  = np.squeeze(train_masks_array[id])
                #print('id = ' + str(id))
                #print('[Before Augmentation] img shape = '  + str(img.shape))             # (256, 400)
                #print('[Before Augmentation] mask shape = ' + str(mask.shape))            # (256, 400)
                ### Extend img/mask to 3d array
                img = img[...,np.newaxis]
                mask = mask[...,np.newaxis]
                ### mask = np.expand_dims(mask, axis=2)
                #print('[Before Augmentation] img shape = '  + str(img.shape))     # (256, 400, 1)
                #print('[Before Augmentation] mask shape = ' + str(mask.shape))    # (256, 400, 1)                
                ### batch is a list
                x_batch.append(img)
                y_batch.append(mask)
            ### Convert batch to 4-d array
            ### 'images' should be either a 4D numpy array of shape (N, height, width, channels)
            ### or a list of 3D numpy arrays, each having shape (height, width, channels).
            ### Grayscale images must have shape (height, width, 1) each.
            ### All images must have numpy's dtype uint8. Values are expected to be in range 0-255.                
            x_batch = np.array(x_batch, np.uint8)       
            y_batch = np.array(y_batch, np.uint8)       
            #print('[INFO] x_batch shape : ' + str(x_batch.shape))      # (2, 256, 400, 1)
            #print('[INFO] y_batch shape : ' + str(y_batch.shape))      # (2, 256, 400, 1)
            ### 1. APPLIED TO Images ONLY: Augment Images ONLY
            ### (image pixelwise Operations ONLY)
            x_batch  = seq_img_only.augment_images(x_batch)
            ### 2. APPLIED TO BOTH Images and Masks:
            ### Affine transformation to Augmentate images & Masks in the SAME way
            #### Augment two batches of images in exactly the SAME way
            #### (e.g. horizontally flip 1st, 2nd and 5th images in both batches, but do not alter 3rd and 4th images)
            ###(flips, crops, and affine) (https://github.com/aleju/imgaug)
            x_batch  = seq_det_both.augment_images(x_batch)
            y_batch  = seq_det_both.augment_images(y_batch)
            #print('[After Augmentation] x_batch shape : ' + str(x_batch.shape))      # (2, 256, 400, 1)
            #print('[After Augmentation] y_batch shape : ' + str(y_batch.shape))      # (2, 256, 400, 1)
            #print('[After Augmentation] x_batch dtype : ' + str(x_batch.dtype))      # uint8
            #print('[After Augmentation] y_batch dtype : ' + str(y_batch.dtype))      # uint8          
            #print("[After Augmentation] x_batch max:"     + str(np.max(x_batch)))    # 255
            #print("[After Augmentation] x_batch min:"     + str(np.min(x_batch)))    # 0
            #print("[After Augmentation] y_batch max:"     + str(np.max(y_batch)))    # 255
            #print("[After Augmentation] y_batch min:"     + str(np.min(y_batch)))    # 0
            ### Convert to float type
            x_batch = x_batch.astype('float32')
            y_batch = y_batch.astype('float32')
            ### Normalize imgs
            x_batch -= mean
            x_batch /= std           
            ### Normalize masks
            y_batch /=255.
            ### yield a batch of imgs and masks
            yield x_batch, y_batch
            
##### Define a val generator
def val_generator(val_imgs_array,val_masks_array,batch_size):
    while True:
##        ## Random seed for data generation
##        seed_gen=np.random.randint(10000,size=1)[0]
##        #print('[Data Generation] seed_gen = ' + str(seed_gen))
##        ## random shuffle data
##        np.random.seed(seed=seed_gen)
##        np.random.shuffle(val_imgs_array)
##        ## random shuffle labels
##        np.random.seed(seed=seed_gen)
##        np.random.shuffle(val_masks_array)  

        ## This is the start of each batch
        for start in range(0, len(val_imgs_array), batch_size):
            ## Initialize a batch of images and masks
            ## The target is to generate a batch and then yield. 
            x_batch = []
            y_batch = []
            
            ## This is the end of the batch
            end = min(start + batch_size, len(val_imgs_array))

            ## img & masks indices in the current batch
            ids_batch = [i for i in range(start,end)]

##            ## Convert the stochastic sequence of augmenters to a deterministic one.
##            ## The deterministic sequence will always apply the exactly same effects to the images.
##            ## NOTE: call this for each batch again, NOT only once at the start
##            seq_det_both = seq_both.to_deterministic() 

            ## Load the batch of images and masks
            for id in ids_batch:                                       
                ### img & mask
                img   = np.squeeze(val_imgs_array[id])
                mask  = np.squeeze(val_masks_array[id])
                #print('id = ' + str(id))
                #print('[Before Augmentation] img shape = '  + str(img.shape))             # (256, 400)
                #print('[Before Augmentation] mask shape = ' + str(mask.shape))            # (256, 400)

                ### Extend img/mask to 3d array
                img = img[...,np.newaxis]
                mask = mask[...,np.newaxis]
                ### mask = np.expand_dims(mask, axis=2)
                #print('[Before Augmentation] img shape = '  + str(img.shape))     # (256, 400, 1)
                #print('[Before Augmentation] mask shape = ' + str(mask.shape))    # (256, 400, 1)
                
                ### batch is a list
                x_batch.append(img)
                y_batch.append(mask)
            ### Convert batch to 4-d array
            ### 'images' should be either a 4D numpy array of shape (N, height, width, channels)
            ### or a list of 3D numpy arrays, each having shape (height, width, channels).
            ### Grayscale images must have shape (height, width, 1) each.
            ### All images must have numpy's dtype uint8. Values are expected to be in range 0-255.                
            x_batch = np.array(x_batch, np.uint8)       
            y_batch = np.array(y_batch, np.uint8)       
            #print('[INFO] x_batch shape : ' + str(x_batch.shape))      # (2, 256, 400, 1)
            #print('[INFO] y_batch shape : ' + str(y_batch.shape))      # (2, 256, 400, 1)
##            ### APPLIED TO Images ONLY: Augment Images ONLY
##            ### (image pixelwise Operations ONLY)
##            x_batch  = seq_img_only.augment_images(x_batch)
##            ### APPLIED TO BOTH Images and Masks: Affine transformation to Augmentate images & Masks in the SAME way
##            #### Augment two batches of images in exactly the SAME way
##            #### (e.g. horizontally flip 1st, 2nd and 5th images in both batches, but do not alter 3rd and 4th images)
##            ###(flips, crops, and affine) (https://github.com/aleju/imgaug)
##            x_batch  = seq_det_both.augment_images(x_batch)
##            y_batch  = seq_det_both.augment_images(y_batch)
##            #print('[After Augmentation] x_batch shape : ' + str(x_batch.shape))      # (2, 256, 400, 1)
##            #print('[After Augmentation] y_batch shape : ' + str(y_batch.shape))      # (2, 256, 400, 1)
##            #print('[After Augmentation] x_batch dtype : ' + str(x_batch.dtype))      # uint8
##            #print('[After Augmentation] y_batch dtype : ' + str(y_batch.dtype))      # uint8          
##            #print("[After Augmentation] x_batch max:"     + str(np.max(x_batch)))    # 255
##            #print("[After Augmentation] x_batch min:"     + str(np.min(x_batch)))    # 0
##            #print("[After Augmentation] y_batch max:"     + str(np.max(y_batch)))    # 255
##            #print("[After Augmentation] y_batch min:"     + str(np.min(y_batch)))    # 0
            ### Convert to float type
            x_batch = x_batch.astype('float32')
            y_batch = y_batch.astype('float32')
            ### Normalize imgs
            x_batch -= mean
            x_batch /= std            
            ### Normalize masks
            y_batch /=255.
            ### yield a batch of imgs and masks
            yield x_batch, y_batch