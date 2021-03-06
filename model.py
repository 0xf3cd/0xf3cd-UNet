import os
import numpy as np
import cv2
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow as tf
from glob import glob
from process_pic import adjust_data
from load_save_pic import test_load_image, save_result, add_suffix
import tensorflow_addons as tfa
from tensorflow.keras import backend as K


# Loss Function
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


# use ImageDataGenerator in Keras to generate more data
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
        yield (img,mask)

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
        seed = seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed
    )

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)
        yield ([img[0]], [mask[0]])
        yield ([img[1]], [mask[1]])


def test_generator(test_files, target_size=(256,256)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)



# Attention U-Net
# From: https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
# Implementation of the paper: https://arxiv.org/abs/1804.03999
def attention_block_2d(x, g, filters, activation='sigmoid', data_format='channels_first'):
    theta_x = Conv2D(filters, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(filters, [1, 1], strides=[1, 1], data_format=data_format)(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation(activation)(psi_f)
    att_x = multiply([x, rate])

    return att_x

def attention_up_and_concate(down_layer, layer, activation='sigmoid', data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
    layer = attention_block_2d(x=layer, g=up, filters=in_channel//4, activation=activation, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


# Octave Convolution
# From: https://github.com/titu1994/keras-octconv/blob/master/octave_conv.py
# Implementation of the paper: https://arxiv.org/abs/1904.05049
def initial_octconv(ip, filters, kernel_size=(3, 3), strides=(1, 1),
                    alpha=0.5, padding='same', dilation=None, bias=False):
    """
    Initializes the Octave Convolution architecture.
    Accepts a single input tensor, and returns a pair of tensors.
    The first tensor is the high frequency pathway.
    The second tensor is the low frequency pathway.
    # Arguments:
        ip: keras tensor.
        filters: number of filters in conv layer.
        kernel_size: conv kernel size.
        strides: strides of the conv.
        alpha: float between [0, 1]. Defines the ratio of filters
            allocated to the high frequency and low frequency
            branches of the octave conv.
        padding: padding mode.
        dilation: dilation conv kernel.
        bias: bool, whether to use bias or not.
    # Returns:
        a pair of tensors:
            - x_high: high frequency pathway.
            - x_low: low frequency pathway.
    """
    if dilation is None:
        dilation = (1, 1)

    high_low_filters = int(alpha * filters)
    high_high_filters = filters - high_low_filters

    if strides[0] > 1:
        ip = AveragePooling2D()(ip)

    # High path
    x_high = Conv2D(high_high_filters, kernel_size, padding=padding,
                    dilation_rate=dilation, use_bias=bias,
                    kernel_initializer='he_normal')(ip)

    # Low path
    x_high_low = AveragePooling2D()(ip)
    x_low = Conv2D(high_low_filters, kernel_size, padding=padding,
                   dilation_rate=dilation, use_bias=bias,
                   kernel_initializer='he_normal')(x_high_low)

    return x_high, x_low

def final_octconv(ip_high, ip_low, filters, kernel_size=(3, 3), strides=(1, 1),
                  padding='same', dilation=None, bias=False):
    """
    Ends the Octave Convolution architecture.
    Accepts two input tensors, and returns a single output tensor.
    The first input tensor is the high frequency pathway.
    The second input tensor is the low frequency pathway.
    # Arguments:
        ip_high: keras tensor.
        ip_low: keras tensor.
        filters: number of filters in conv layer.
        kernel_size: conv kernel size.
        strides: strides of the conv.
        padding: padding mode.
        dilation: dilation conv kernel.
        bias: bool, whether to use bias or not.
    # Returns:
        a single Keras tensor:
            - x_high: The merged high frequency pathway.
    """
    if dilation is None:
        dilation = (1, 1)

    if strides[0] > 1:
        avg_pool = AveragePooling2D()

        ip_high = avg_pool(ip_high)
        ip_low = avg_pool(ip_low)

    # High path
    x_high_high = Conv2D(filters, kernel_size, padding=padding,
                         dilation_rate=dilation, use_bias=bias,
                         kernel_initializer='he_normal')(ip_high)

    # Low path
    x_low_high = Conv2D(filters, kernel_size, padding=padding,
                        dilation_rate=dilation, use_bias=bias,
                        kernel_initializer='he_normal')(ip_low)

    x_low_high = UpSampling2D(interpolation='nearest')(x_low_high)

    # Merge paths
    x = add([x_high_high, x_low_high])

    return x

def octconv_block(ip_high, ip_low, filters, kernel_size=(3, 3), strides=(1, 1),
                  alpha=0.5, padding='same', dilation=None, bias=False):
    """
    Constructs an Octave Convolution block.
    Accepts a pair of input tensors, and returns a pair of tensors.
    The first tensor is the high frequency pathway for both ip/op.
    The second tensor is the low frequency pathway for both ip/op.
    # Arguments:
        ip_high: keras tensor.
        ip_low: keras tensor.
        filters: number of filters in conv layer.
        kernel_size: conv kernel size.
        strides: strides of the conv.
        alpha: float between [0, 1]. Defines the ratio of filters
            allocated to the high frequency and low frequency
            branches of the octave conv.
        padding: padding mode.
        dilation: dilation conv kernel.
        bias: bool, whether to use bias or not.
    # Returns:
        a pair of tensors:
            - x_high: high frequency pathway.
            - x_low: low frequency pathway.
    """
    if dilation is None:
        dilation = (1, 1)

    low_low_filters = high_low_filters = int(alpha * filters)
    high_high_filters = low_high_filters = filters - low_low_filters

    avg_pool = AveragePooling2D()

    if strides[0] > 1:
        ip_high = avg_pool(ip_high)
        ip_low = avg_pool(ip_low)

    # High path
    x_high_high = Conv2D(high_high_filters, kernel_size, padding=padding,
                         dilation_rate=dilation, use_bias=bias,
                         kernel_initializer='he_normal')(ip_high)

    x_low_high = Conv2D(low_high_filters, kernel_size, padding=padding,
                        dilation_rate=dilation, use_bias=bias,
                        kernel_initializer='he_normal')(ip_low)
    x_low_high = UpSampling2D(interpolation='nearest')(x_low_high)

    # Low path
    x_low_low = Conv2D(low_low_filters, kernel_size, padding=padding,
                       dilation_rate=dilation, use_bias=bias,
                       kernel_initializer='he_normal')(ip_low)

    x_high_low = avg_pool(ip_high)
    x_high_low = Conv2D(high_low_filters, kernel_size, padding=padding,
                        dilation_rate=dilation, use_bias=bias,
                        kernel_initializer='he_normal')(x_high_low)

    # Merge paths
    x_high = add([x_high_high, x_low_high])
    x_low = add([x_low_low, x_high_low])

    return x_high, x_low

def my_Conv2D(prev_layer, origin_arg, 
              activation='relu', padding='same', 
              dropout_rate=0.0, perf_BN=False, perf_IN=False):
    new_layer = Conv2D(*origin_arg, activation=activation, padding=padding)(prev_layer)
    if dropout_rate > 0.0:
        new_layer = Dropout(dropout_rate)(new_layer)
    if perf_IN:
        new_layer = tfa.layers.InstanceNormalization(axis=3, 
                                                     center=True, 
                                                     scale=True,
                                                     beta_initializer="random_uniform",
                                                     gamma_initializer="random_uniform")(new_layer)
    if perf_BN:
        new_layer = BatchNormalization()(new_layer)
    return new_layer



# Definition of the model
def origin_unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


def oct_att_unet(input_size=(256,256,1), init_filters=32, 
                 activation='relu', padding='same', 
                 dropout_rate=0.0, perf_BN=False, 
                 data_format='channels_last', final_act='sigmoid', 
                 attention_act='sigmoid', use_attention=True):
    inputs = Input(input_size)
    filters = init_filters
    depth = 4
    convs = []
    pools = [inputs]

    # Down-sampling
    for i in range(depth):
        prev_layer = pools[-1]
        new_conv = my_Conv2D(prev_layer, (filters, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=perf_BN, perf_IN=False)
        new_conv = my_Conv2D(new_conv, (filters, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=perf_BN, perf_IN=False)
        convs.append(new_conv)
        new_pool = MaxPooling2D(pool_size=(2, 2))(new_conv)
        pools.append(new_pool)
        filters *= 2

        # concatenate([, ], axis=3)
    
    # Bottleneck - the bottom of U-Net
    prev_layer = pools[-1]
    xh5, xl5 = initial_octconv(prev_layer, filters=filters)
    xh5, xl5 = octconv_block(xh5, xl5, filters=int(filters*1.5))
    conv_bottleneck = final_octconv(xh5, xl5, filters=filters)
    convs.append(conv_bottleneck)
    filters //= 2

    # Up-sampling
    for i in range(depth):
        last_conv = convs[-1]
        to_concat_conv = convs[depth-1-i]
        if use_attention:  
            up = attention_up_and_concate(last_conv, to_concat_conv, activation=attention_act, data_format=data_format)
        else:
            up = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(last_conv), to_concat_conv], axis=3)
        new_conv = my_Conv2D(up, (filters, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=perf_BN, perf_IN=False)
        new_conv = my_Conv2D(new_conv, (filters, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=perf_BN, perf_IN=False)
        convs.append(new_conv)
        # new_pool = MaxPooling2D(pool_size=(2, 2))(new_conv)
        # layers.append(new_pool)
        filters //= 2

    conv_last = Conv2D(1, (1, 1), activation=final_act)(convs[-1])
    return Model(inputs=[inputs], outputs=[conv_last])



def oct_att_IN_unet(input_size=(256,256,1), init_filters=32, 
                            activation='relu', padding='same', 
                            dropout_rate=0.0,
                            data_format='channels_last', final_act='sigmoid', 
                            attention_act='sigmoid', use_attention=True):
    inputs = Input(input_size)
    filters = init_filters
    depth = 4
    convs = []
    pools = [inputs]

    # Down-sampling
    for i in range(depth):
        prev_layer = pools[-1]
        new_conv_1 = my_Conv2D(prev_layer, (filters//2, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=False, perf_IN=False)
        new_conv_2 = my_Conv2D(prev_layer, (filters//2, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=False, perf_IN=False)
        new_conv = concatenate([new_conv_1, new_conv_2], axis=3)
        new_conv = my_Conv2D(new_conv, (filters, (3, 3)),
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=False, perf_IN=True)
        convs.append(new_conv)
        new_pool = MaxPooling2D(pool_size=(2, 2))(new_conv)
        pools.append(new_pool)
        filters *= 2

        # concatenate([, ], axis=3)
    
    # Bottleneck - the bottom of U-Net
    prev_layer = pools[-1]
    xh5, xl5 = initial_octconv(prev_layer, filters=filters)
    xh5, xl5 = octconv_block(xh5, xl5, filters=int(filters*1.5))
    conv_bottleneck = final_octconv(xh5, xl5, filters=filters)
    convs.append(conv_bottleneck)
    filters //= 2

    # Up-sampling
    for i in range(depth):
        last_conv = convs[-1]
        to_concat_conv = convs[depth-1-i]
        if use_attention:  
            up = attention_up_and_concate(last_conv, to_concat_conv, activation=attention_act, data_format=data_format)
        else:
            up = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(last_conv), to_concat_conv], axis=3)
        new_conv = my_Conv2D(up, (filters, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=False, perf_IN=False)
        new_conv = my_Conv2D(new_conv, (filters, (3, 3)), 
                             activation=activation, padding=padding,
                             dropout_rate=dropout_rate, perf_BN=False, perf_IN=False)
        convs.append(new_conv)
        # new_pool = MaxPooling2D(pool_size=(2, 2))(new_conv)
        # layers.append(new_pool)
        filters //= 2

    conv_last = Conv2D(1, (1, 1), activation=final_act)(convs[-1])
    return Model(inputs=[inputs], outputs=[conv_last])
