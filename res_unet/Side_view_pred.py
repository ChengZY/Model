#### IME circles

import os
import warnings
import numpy as np
from skimage.io import imsave
from keras import backend as K
import U_net_model
import Image_preprocess
from skimage.transform import resize

# Image Format 'channels_last'
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
# Set Random Seed
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
SEED = 42
### Raw Image Dimensions
### Raw image have different dimensions
Raw_channels = 1
### Resized Image dimension
image_rows = 256
image_cols = 400
image_channels = 1
# Binary Mask
num_classes = 1
# threshold
threshold = 0.5
# Epochs & minibatch size
epochs = 100
batch_size = 2
k=16

#### Define a combined data generator
#### function to combine generators (https://github.com/keras-team/keras/issues/5720)
def combine_generator(gen1, gen2):
    while True:  ### BE SURE TO add 'True' to make it a loop
        yield (gen1.next(), gen2.next())  ### yield will make this function a generator and iteratively produce next value without end, NOT the same batch.

### ------------Path of the test data --------------------------------------------------------------
#test_imgs_path   =     '../Ground_Truth/Images/'  
test_imgs_path   =      '../data_2/train/Images'
pred_path        =     ['./PredMasks/2','./PredMasks/3']
model_path       =     ['./Models/unetRes_plate_{}x{}_k{}.h5'.format(image_rows,image_cols,k),\
                        './Models/unetRes_plateo_{}x{}_k{}.h5'.format(image_rows,image_cols,k)]
# ---------------------------------------------------------------------------------------------
### ------------------------------Load & Display and normaalization testing images-----------------------
# list of file names including suffix

def prepare_test_data(test_imgs_path):
    test_imgs_array, sizes = Image_preprocess.load_img(test_imgs_path)
    test_imgs_array = Image_preprocess.Normalization_img(test_imgs_array)
    files = os.listdir(test_imgs_path)
    return test_imgs_array,files
    
def predict_component(test_imgs_array,files, pred_path , model_path):    
    ### ----------------------------------------------------Build U-Net model----------------------------------------------------------
    model = U_net_model.UnetResidual_model(input_shape=(image_rows, image_cols, image_channels), num_classes=1, k=16)   
    model.load_weights(model_path)
    preds_test_mask = model.predict(test_imgs_array)    
    print('-' * 70)
    print('Saving predicted test masks to files...')
    print('-' * 70)    
    model = U_net_model.UnetResidual_model(input_shape=(image_rows,image_cols,image_channels),num_classes=1,  k=16)
    i = 0
    for filename in files:
        print(filename)
        image_file_name = filename.split('.')[0] + '.jpg'
        full_mask_name = os.path.join(pred_path, image_file_name)
        print(full_mask_name)
        ### raw size predicted mask
        prob = np.squeeze(preds_test_mask[i])
        prob = (prob > threshold).astype('float32')
        mask = (prob * 255.).astype(np.uint8) 
        #mask = resize(mask, (2048, 2048))
        imsave(full_mask_name, mask)
        i = i + 1


test_imgs_array,files = prepare_test_data(test_imgs_path)

for i in range(2):
    predict_component(test_imgs_array,files, pred_path[i] , model_path[i])
