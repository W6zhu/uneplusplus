import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
import os
import io
import random
import nibabel
import numpy as np
import nibabel as nib
from nibabel import load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
import tensorflow as tf
import tifffile as tiff
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import random
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Concatenate
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
import os
import io
import random
import nibabel
import numpy as np
from glob import glob
import nibabel as nib
import tensorflow as tf
from nibabel import load
import matplotlib.pyplot as plt
from keras.utils import Sequence
from IPython.display import Image, display
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import layers
from keras.layers import Input, concatenate, UpSampling2D,BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.transform import resize
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        BatchNormalization,
        Conv3D,
        Conv3DTranspose,
        MaxPooling3D,
        Dropout,
        SpatialDropout3D,
        UpSampling3D,
        Input,
        concatenate,
        multiply,
        add,
        Activation,
    )
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float64)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float64)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def tprf(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold)
   
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))

    if (tp == 0):
        tpr = 0
    else:
        tpr = tp / (tp + fn)

    return tpr

def fprf(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold)

    fp = np.sum((y_pred == 1) & (y_true == 0))

    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    if (fp == 0):
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    return fpr


################################################################################################################################

def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)
    x = Dropout(0.1)(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x

def simple_unet_plus_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path (Encoder)
    c1 = conv_block(s, 16)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 128)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 256)

    # Expansive path with nested skip pathways (Decoder)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 128)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3, Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)])
    c7 = conv_block(u7, 64)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2, Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3), Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(c4)])
    c8 = conv_block(u8, 32)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1, Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2), Conv2DTranspose(16, (3, 3), strides=(4, 4), padding='same')(c3), Conv2DTranspose(16, (4, 4), strides=(8, 8), padding='same')(c4)])
    c9 = conv_block(u9, 16)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    model.summary()


    return model


################################################################################################################################

def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = tf.keras.backend.flatten(y_true)
  y_pred_f = tf.keras.backend.flatten(y_pred)
  intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def tprf(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold)
   
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))

    if (tp == 0):
        tpr = 0
    else:
        tpr = tp / (tp + fn)

    return tpr

def fprf(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold)

    fp = np.sum((y_pred == 1) & (y_true == 0))

    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    if (fp == 0):
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    return fpr

################################################################################################################################

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='bce', metrics=[dice_coef])
    model.summary()
    
    return model

###########################################################################################################################

def encoder_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(input_tensor, skip_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = UpSampling2D(size=(2, 2))(input_tensor)
    x = Concatenate()([x, skip_tensor])
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_segnet_model(input_img, n_filters=64, n_classes=1, dropout=0.1, batchnorm=True):
    # Contracting Path (encoder)
    c1 = encoder_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = encoder_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = encoder_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = encoder_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Expanding Path (decoder)
    u6 = decoder_block(c4, c3, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    u7 = decoder_block(u6, c2, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    u8 = decoder_block(u7, c1, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    
    # Output layer
    output_img = Conv2D(n_classes, (1, 1), activation='sigmoid')(u8)
    
    return Model(inputs=input_img, outputs=output_img)


from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)

def calculate_tpr_fpr(y_true, y_pred):
    # Assuming y_pred is sigmoid output, threshold to get binary mask
    y_pred = y_pred > 0.5
    # Flatten the arrays to compute confusion matrix
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_f).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 30
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

lr_scheduler = LearningRateScheduler(lr_scheduler, verbose=1)

# Model setup and training
input_img = Input((224, 224, 1), name='img')

###########################################################################################################################


image_directory = 'MRI19/Anatomical_mag_echo5/'
mask_directory = 'MRI19/whole_liver_segmentation/'

image_dataset = []  
mask_dataset = []
sliced_image_dataset = []
sliced_mask_dataset = []

# SIZE = 128

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(image_directory+image_name)
        image = np.array(image.get_fdata())
        #image = resize(image, (SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'nii'):
        image = nib.load(mask_directory+image_name)
        image = np.array(image.get_fdata())
        #image = resize(image, (SIZE, SIZE))
        mask_dataset.append(np.array(image))

for i in range(len(image_dataset)):
    print(image_dataset[i].shape[2])
    for j in range(image_dataset[i].shape[2]):
        sliced_image_dataset.append(image_dataset[i][:,:,j])

for i in range(len(mask_dataset)):
    for j in range(mask_dataset[i].shape[2]):
        if i == 16 and j == 25:
            continue
        else:
            sliced_mask_dataset.append(mask_dataset[i][:,:,j])

#Normalize images
sliced_image_dataset = np.expand_dims(np.array(sliced_image_dataset),3)
#D not normalize masks, just rescale to 0 to 1.
sliced_mask_dataset = np.expand_dims((np.array(sliced_mask_dataset)),3)

X_train, X_test, y_train, y_test = train_test_split(sliced_image_dataset, sliced_mask_dataset, test_size = 0.20, random_state = 0)

##############################################################################################################################################

dice_scores = []
TPRs = []
FPRs = []

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]

def get_model1():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model1 = get_model1()

def get_model2():
    return simple_unet_plus_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model2 = get_model2()

model3 = get_segnet_model(input_img)

##############################################################################################################################################

dice_scores = []
TPRs = []
FPRs = []

IMG_HEIGHT = sliced_image_dataset.shape[1]
IMG_WIDTH  = sliced_image_dataset.shape[2]
IMG_CHANNELS = sliced_image_dataset.shape[3]


model1.load_weights(f'C:/Users/Mittal/Desktop/kunet/best_model4.keras')
model2.load_weights(f'C:/Users/Mittal/Desktop/wunet++/best_model1.keras')
model3.load_weights(f'C:/Users/Mittal/Desktop/tfSegNet/model_checkpoint0.h5')


test_img = sliced_image_dataset[14]
ground_truth = sliced_mask_dataset[14]
test_img_norm = test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction1 = (model1.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
prediction2 = (model2.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
prediction3 = (model3.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)


original_image_normalized1 = ground_truth.astype(float) / np.max(ground_truth)
colored_mask1 = plt.get_cmap('jet')(prediction1 / np.max(prediction1))
alpha = 0.5 
colored_mask1[..., 3] = np.where(prediction1 > 0, alpha, 0)

original_image_normalized2 = ground_truth.astype(float) / np.max(ground_truth)
colored_mask2 = plt.get_cmap('jet')(prediction2 / np.max(prediction2))
alpha = 0.5 
colored_mask2[..., 3] = np.where(prediction2 > 0, alpha, 0)

original_image_normalized3 = ground_truth.astype(float) / np.max(ground_truth)
colored_mask3 = plt.get_cmap('jet')(prediction3 / np.max(prediction3))
alpha = 0.5 
colored_mask3[..., 3] = np.where(prediction3 > 0, alpha, 0)

plt.figure(figsize=(16, 16))

plt.subplot(3,4,1)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.axis('off')

plt.subplot(3,4,2)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.axis('off')

plt.subplot(3,4,3)
plt.title('U-Net Prediction')
plt.imshow(prediction1, cmap='gray')
plt.axis('off')

plt.subplot(3,4,4)
plt.title("Overlayed Images")
plt.imshow(original_image_normalized1, cmap='gray')
plt.imshow(colored_mask1, cmap='jet')
plt.axis('off')

plt.subplot(3,4,5)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.axis('off')

plt.subplot(3,4,6)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.axis('off')

plt.subplot(3,4,7)
plt.title('U-Net++ Prediction')
plt.imshow(prediction2, cmap='gray')
plt.axis('off')

plt.subplot(3,4,8)
plt.title("Overlayed Images")
plt.imshow(original_image_normalized2, cmap='gray')
plt.imshow(colored_mask2, cmap='jet')
plt.axis('off')

plt.subplot(3,4,9)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.axis('off')

plt.subplot(3,4,10)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.axis('off')

plt.subplot(3,4,11)
plt.title('SegNet Prediction')
plt.imshow(prediction3, cmap='gray')
plt.axis('off')

plt.subplot(3,4,12)
plt.title("Overlayed Images")
plt.imshow(original_image_normalized3, cmap='gray')
plt.imshow(colored_mask3, cmap='jet')
plt.axis('off')


plt.savefig(f'C:/Users/Mittal/Desktop/CNN Results/f_2396_15.png')
plt.close()
