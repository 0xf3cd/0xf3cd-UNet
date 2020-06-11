#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow as tf

from glob import glob
from tqdm import tqdm

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizers import Adam

import matplotlib.image as mpimg
import tensorflow.python.keras.backend as K


# In[4]:


def add_colored_dilate(image, mask_image, dilate_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    dilate_image_gray = cv2.cvtColor(dilate_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    dilate = cv2.bitwise_and(dilate_image, dilate_image, mask=dilate_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])
    dilate_coord = np.where(dilate!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    dilate[dilate_coord[0],dilate_coord[1],:] = [0,0,255]

    ret = cv2.addWeighted(image, 0.7, dilate, 0.3, 0)
    ret = cv2.addWeighted(ret, 0.7, mask, 0.3, 0)

    return ret

def add_colored_mask(image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def my_add_colored_mask(image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask_image_gray>[10])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def diff_mask(ref_image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(ref_image, 0.7, mask, 0.3, 0)
    return ret


# In[5]:


# From: https://github.com/zhixuhao/unet/blob/master/data.py
def train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)

def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

# From: https://github.com/zhixuhao/unet/blob/master/data.py
def my_train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (np.array([img[0]]),np.array([mask[0]]))
        yield (np.array([img[1]]),np.array([mask[1]]))


# In[6]:


# From: https://github.com/zhixuhao/unet/blob/master/data.py
def test_load_image(test_file, target_size=(256,256)):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator(test_files, target_size=(256,256)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)
        
def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.).astype(np.uint8)

        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))

        cv2.imwrite(result_file, img)

def add_suffix(base_file, suffix):
    filename, fileext = os.path.splitext(base_file)
    return "%s_%s%s" % (filename, suffix, fileext)


# In[7]:


# From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def mean_iou(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return ((intersection + K.epsilon()) / (union + K.epsilon()))

def mean_iou_loss(y_true, y_pred):
    return -mean_iou(y_true, y_pred)

# In[8]:


init_learning_rate = 1e-5 # do not change
pic_size = 512 # do not change
test_files = [test_file for test_file in glob(os.path.join('./test/', "*.png"))               if ("_mask" not in test_file                   and "_dilate" not in test_file                   and "_predict" not in test_file)]


# In[10]:


def load_trained_model(mdir, show_summary=True):
    h5_dir = os.path.join(mdir, 'unet_64.hdf5')
    model = load_model(h5_dir, custom_objects={"tf": tf}, compile=False)
#     model = load_model(h5_dir, custom_objects={"tf": tf})
    if show_summary:
        model.summary()
    model.compile(optimizer=Adam(lr=init_learning_rate, clipnorm=1.), loss=mean_iou_loss, metrics=[dice_coef, mean_iou, 'binary_accuracy'])
    return model


# In[11]:


def get_test_data(togray=True):
    test_set = [[[0]*512]*512]*len(test_files)
    for i in range(len(test_files)):
        tdir = test_files[i]
        test_pic = cv2.imread(os.path.join(tdir))
#         test_pic = test_load_image(tdir, (512, 512))
        if togray:
            test_pic = cv2.cvtColor(test_pic, cv2.COLOR_RGB2GRAY)
        test_set[i] = test_pic
    return np.array(test_set)

def get_mask_data(togray=True):
    mask_set = [[[0]*512]*512]*len(test_files)
    for i in range(len(test_files)):
        tdir = test_files[i]
        tdir = tdir.split('.')
        mask_pic = cv2.imread(os.path.join('.'+tdir[1]+'_mask.'+tdir[2]))
#         print(os.path.join(tdir[0]+tdir[1]+'_mask.'+tdir[2]))
        if togray:
            mask_pic = cv2.cvtColor(mask_pic, cv2.COLOR_RGB2GRAY)
        mask_set[i] = mask_pic
    return np.array(mask_set)

def get_pred_results(model):
    test_gen = test_generator(test_files, target_size=(pic_size,pic_size))
    results = model.predict_generator(test_gen, len(test_files), verbose=1)
    return results[:,:,:,0]


# In[12]:


def save_pred_results(mdir, pred_results, fname='test_res'):
    model_2_pred_save_dir = os.path.join(mdir, fname)
    if not os.path.exists(model_2_pred_save_dir):
        os.makedirs(model_2_pred_save_dir)
    
    for i in range(len(test_files)):
        tdir = test_files[i]

        os.system('cp '+tdir+' '+model_2_pred_save_dir)

        tdir = tdir.split('/')
        res_pic = pred_results[i].copy()
        res_pic *= 255
        cv2.imwrite(os.path.join(model_2_pred_save_dir, tdir[-1]+'_pred.png'), res_pic)

def load_prev_pred_results(mdir, fname='test_res', togray=True):
    model_2_pred_save_dir = os.path.join(mdir, fname)
    results = []
    for i in range(len(test_files)):
        tdir = test_files[i]
        tdir = tdir.split('/')
        readin = cv2.imread(os.path.join(model_2_pred_save_dir, tdir[-1]+'_pred.png'))
        if togray:
            readin = cv2.cvtColor(readin, cv2.COLOR_RGB2GRAY)
        results.append(readin)
    return np.array(results)
        
# In[14]:


def show_acc_loss_dice_iou(mdir, num=10):
    training_loss = np.load(os.path.join(mdir, 'training_loss.npy'))
    validation_loss = np.load(os.path.join(mdir, 'validation_loss.npy'))
    training_accuracy = np.load(os.path.join(mdir, 'training_accuracy.npy'))
    validation_accuracy = np.load(os.path.join(mdir, 'validation_accuracy.npy'))
    dice_coef_value = np.load(os.path.join(mdir, 'dice_coef.npy'))
    mean_iou_value = np.load(os.path.join(mdir, 'mean_iou.npy'))
    
    values = dict(
        training_loss=training_loss[-num:],
        validation_loss=validation_loss[-num:],
        training_accuracy=training_accuracy[-num:],
        validation_accuracy=validation_accuracy[-num:],
        dice_coef_value=dice_coef_value[-num:],
        mean_iou_value=mean_iou_value[-num:]
    )
    print(values)
    

def show_acc_loss_dice_iou_fig(mdir, save_fig_fname='evaluate', figsize=(15, 15)):
    fig, axs = plt.subplots(4, 1, figsize=figsize)

    training_loss = np.load(os.path.join(mdir, 'training_loss.npy'))
    validation_loss = np.load(os.path.join(mdir, 'validation_loss.npy'))
    training_accuracy = np.load(os.path.join(mdir, 'training_accuracy.npy'))
    validation_accuracy = np.load(os.path.join(mdir, 'validation_accuracy.npy'))
    dice_coef_value = np.load(os.path.join(mdir, 'dice_coef.npy'))
    mean_iou_value = np.load(os.path.join(mdir, 'mean_iou.npy'))

    epoch_count = range(1, len(training_loss) + 1)

    axs[0].plot(epoch_count, training_loss, 'r--')
    axs[0].plot(epoch_count, validation_loss, 'b-')
    axs[0].legend(['Training Loss', 'Validation Loss'])

    axs[1].plot(epoch_count, training_accuracy, 'r--')
    axs[1].plot(epoch_count, validation_accuracy, 'b-')
    axs[1].legend(['Training Accuracy', 'Validation Accuracy'])

    axs[2].plot(epoch_count, dice_coef_value, 'r--')
    axs[2].legend(['Dice Coef'])

    axs[3].plot(epoch_count, mean_iou_value, 'r--')
    axs[3].legend(['Mean IOU'])
    
    if save_fig_fname != None:
        axs[0].get_figure().savefig(os.path.join(mdir, save_fig_fname))


# In[15]:


def show_test_pic(mdir, pic_name='test'):
    img = mpimg.imread(os.path.join(mdir, pic_name+'.png'))
#     img0 = mpimg.imread(os.path.join(mdir, pic_name+'0.png'))
#     img1 = mpimg.imread(os.path.join(mdir, pic_name+'1.png'))

#     fig = plt.figure(figsize=(20, 10))  
#     ax1 = fig.add_subplot(1, 2, 1)  
#     ax1.imshow(img0, cmap=plt.cm.gray)  
#     ax2 = fig.add_subplot(1, 2, 2)  
#     ax2.imshow(img1, cmap=plt.cm.gray)  
#     ax1.axis('off')
#     ax2.axis('off')
    fig = plt.figure(figsize=(10, 10)) 
    ax1 = fig.add_subplot(1, 1, 1)  
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.axis('off')
    plt.show()


# In[17]:


def get_layer_info(model):
    layers = model.layers
    layer_info = []
#     print(layers)
#     print(layer_info)
    for l in layers:
        layer_info.append(dict(
            layer=l,
            name=l.name
        ))
    return layer_info

def get_layer_output(model, input_pic, adjust_255=True):
    input_pic_ = input_pic.copy()
    input_pic_ = input_pic_.reshape((512, 512, 1))
    input_pic_ = np.array([input_pic_])
#     print(input_pic_.shape)
    layer_info = get_layer_info(model)
    for idx in range(1, len(layer_info)):
        get_mid_output = K.function([model.layers[0].input],  [model.layers[idx].output])
        mid_output = get_mid_output([input_pic_])
        if adjust_255:
            mid_output *= 255
        layer_info[idx]['output'] = mid_output[0][0,:,:,0]
    return layer_info

def save_layer_output(mdir, fname, layer_info):
    layer_num = len(layer_info)
    root_dir = os.path.join(mdir, 'layer_middle_output')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for idx in range(1, layer_num):
        save_fdir = os.path.join(root_dir, str(idx)+layer_info[idx]['name'])
        if not os.path.exists(save_fdir):
            os.makedirs(save_fdir)
            # raise RuntimeError('FolderExisted: ' + save_fdir)
        
        cv2.imwrite(os.path.join(save_fdir, fname), layer_info[idx]['output']*255)
        

# In[18]:


def show_pic(pic, adjust_255=False):
    if adjust_255:
        pic *= 255
    fig = plt.figure(figsize=(20, 10))  
    ax1 = fig.add_subplot(1, 1, 1)  
    ax1.imshow(pic, cmap=plt.cm.gray)  
    ax1.axis('off')
    plt.show()


# In[ ]:




