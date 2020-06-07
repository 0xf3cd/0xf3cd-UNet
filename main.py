import os
from glob import glob
import numpy as np
import time
from model import *

# these paths are also used in process_dataset.py
INPUT_DIR = os.path.join(".", "input")

SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentation")
SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
SEGMENTATION_AUG_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "augmentation")
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")
SEGMENTATION_DILATE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "dilate")
SEGMENTATION_SOURCE_DIR = os.path.join(INPUT_DIR, "pulmonary-chest-xray-abnormalities")

train_files = glob(os.path.join(SEGMENTATION_IMAGE_DIR, "*.png"))
test_files = glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png"))
mask_files = glob(os.path.join(SEGMENTATION_MASK_DIR, "*.png"))
dilate_files = glob(os.path.join(SEGMENTATION_DILATE_DIR, "*.png"))

print(len(train_files),  len(test_files),  len(mask_files),  len(dilate_files))

pic_size = 512 # unchange
init_learning_rate = 1e-5 # unchange
init_filters = 64 # unchange
epochs = 512 
steps_per_epoch = 16
batch_size = steps_per_epoch

test_files = [test_file for test_file in glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png")) \
              if ("_mask" not in test_file \
                          and "_dilate" not in test_file \
                          and "_predict" not in test_file)]

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_gen = train_generator(batch_size,
                            SEGMENTATION_TRAIN_DIR,
                            'image',
                            'dilate', 
                            train_generator_args,
                            target_size=(pic_size,pic_size),
                            save_to_dir=os.path.abspath(SEGMENTATION_AUG_DIR))

validation_data = (test_load_image(test_files[0], target_size=(pic_size, pic_size)),
                test_load_image(add_suffix(test_files[0], "dilate"), target_size=(pic_size, pic_size)))

model = octave_attention_unet(input_size=(pic_size,pic_size,1), init_filters=init_filters, 
                                activation='relu', padding='same', 
                                dropout_rate=0.3, perf_BN=False, 
                                data_format='channels_last', final_act='sigmoid',
                                attention_act='sigmoid', use_attention=True)
model.compile(optimizer=Adam(lr=init_learning_rate, clipnorm=1.), \
                             loss=dice_coef_loss, \
                             metrics=[dice_coef, 'binary_accuracy'])
model.summary()

save_fdir = './octatt_unet_64'+time.strftime('%Y-%m-%d-%h-%s',time.localtime(time.time()))+'/'
if not os.path.exists(save_fdir):
    os.makedirs(save_fdir)
    
early_stopping = EarlyStopping(monitor='val_loss', patience=1000, verbose=1)
model_checkpoint = ModelCheckpoint(save_fdir+'unet_64.hdf5', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True)

class TrainRec(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        test_file_dir0 = test_files[0]        
        test0 = test_load_image(test_file_dir0, (512, 512))
        
        self.count = 0
        self.save_fdir = save_fdir
        self.test0 = test0.copy()
        
        test0 = test0.reshape((512, 512))
        test0 = test0 * 255
        cv2.imwrite(self.save_fdir+'test.png', test0)

    def on_epoch_end(self, epoch, logs={}):
        pred0 = model.predict([self.test0])
        pred0 = pred0[0]
        pred0 = pred0.reshape((512, 512))
        pred0 = pred0 * 255
        cv2.imwrite(self.save_fdir+str(self.count)+'.png', pred0)
        self.count += 1
train_rec = TrainRec()

octave_attention_unet_history = model.fit_generator(train_gen,
                                steps_per_epoch=steps_per_epoch, 
                                epochs=epochs, 
                                callbacks=[early_stopping, model_checkpoint, train_rec],
                                validation_data = validation_data)

history = octave_attention_unet_history
np.save(save_fdir+'training_loss', history.history['loss'])
np.save(save_fdir+'validation_loss', history.history['val_loss'])
np.save(save_fdir+'dice_coef', history.history['dice_coef'])
np.save(save_fdir+'training_accuracy', history.history['binary_accuracy'])
np.save(save_fdir+'validation_accuracy', history.history['val_binary_accuracy'])

model.save(save_fdir+'model.h5')
model.save_weights(save_fdir+'model_weights.h5')

# TODO: Now, if the model is run directly, then Keras will raise an error called keras_scratch_graph.
# If the original U-Net is previously run, then the octave_attention_unet can be run correctly.
# There might be something wrong with the structure of the network oe the train_generator, which needs to be fixed