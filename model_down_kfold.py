#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  # plt ?�於顯示?��?
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from PIL import Image
import efficientnet.keras as efn 
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=2,
                        verbose=1,
                        mode='max',
                        epsilon=0.0001)

#linux-------
base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
trainpath = base_path + '/train'
val_path = base_path + '/validation'
pltsave_path = base_path+'/plt_img'

# Model -----------------------------------------------------------------------
#model_net = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# model_net = efn.EfficientNetL2(input_shape=(128,128,3), # 當 include_top=False 時，可調整輸入圖片的尺寸（長寬需不小於 32）
#   weights="./efficientnet-l2_noisy-student_notop.h5", 
#   #weights='imagenet',
#   include_top=False,# 是否包含最後的全連接層 (fully-connected layer)
#   drop_connect_rate=0,  # the hack
#   pooling='avg'# 當 include_top=False 時，最後的輸出是否 pooling（可選 'avg' 或 'max'）
# )
model_net = tf.keras.applications.EfficientNetB7(input_shape=(128,128,3),weights="imagenet",include_top=False, pooling='avg',classifier_activation="softmax")
#freeze some layers
#for layer in model_net.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    #layer.trainable = False
#model_net.trainable = False

#build the category classification branch in the model
x = model_net.output
x = layers.Dropout(0.5)(x)
x1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
y1 = Dense(4, activation='softmax', name='category')(x1)

#create final model by specifying the input and outputs for the branches
final_model = Model(inputs=model_net.input, outputs=y1)

#print(final_model.summary())

#opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
opt = Adam(learning_rate=0.001)

final_model.compile(optimizer=opt,loss={'category':'categorical_crossentropy'
                                        },
                    metrics={'category':['accuracy'] 
         }
                    ) #default:top-5

# Loading the data-------------------------------------------------------------

train_datagen = ImageDataGenerator(#rotation_range=30.,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True,
                                    brightness_range= [0.8, 1.2],
                                    rotation_range = 10,
                                    channel_shift_range=100,
                                    #preprocessing_function = myFunc
                                    vertical_flip = True
                                    )

test_datagen = ImageDataGenerator()

def generate_arrays_from_file(trainpath,set_len,file_nums,has_remainder=0,batch_size=32):
    
    cnt = 0 
    pos = 0
    inputs = None
    labels_category = None

    while 1:
        if cnt % (set_len // batch_size+has_remainder) == 0:  #?�斷?�否讀完�??�個�?�?
            pos = 0
            seq = cnt // (set_len // batch_size + has_remainder) % file_nums #此次讀?�第seq?��?�?
            del inputs, labels_category
            inputs = np.load(os.path.join(trainpath, 'inputs' + str(seq + 1)+ '0420_train_sleeve.npy'))
            labels_category = np.load(os.path.join(trainpath, 'labels' + str(seq + 1)+ '0420_train_sleeve.npy'))
        print("---generate trainfile arrays",seq,"--", inputs.shape)
        start = pos*batch_size
        end = min((pos+1)*batch_size, set_len-1)
        batch_inputs = inputs[start:end]
        batch_labels_category = labels_category[start:end]
        pos += 1
        cnt += 1
        print("batch label shape ",batch_labels_category.shape)
        yield (batch_inputs, batch_labels_category)

# 設�?超�??�HyperParameters
epochs = 30
batch = 32 #128
file_number = 1
#file_len = 3776#21600
train_data = np.load(os.path.join(trainpath,'inputs1train_down_0602.npy'))
train_target = np.load(os.path.join(trainpath,'labels1train_down_0602.npy'))
print(len(train_data),len(train_target))
k = 5
num_val_samples = len(train_data) // k
train_acc_list = []
val_acc_list = []
# print(len(x_data),len(y_data))
# datasize = int(len(x_data)*0.8)
# x_train = x_data[:datasize]
# y_train = y_data[:datasize]
# datasize = int(len(x_data)*0.2)
# x_val = x_data[-datasize:]
# y_val_category = y_data[-datasize:]
# print(len(x_train),len(y_train),len(x_val),len(y_val_category))
for i in range(k):
    print("preprocessing fold #",i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_target = train_target[i * num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]], axis = 0)
    partial_train_target = np.concatenate([train_target[:i * num_val_samples], train_target[(i+1) * num_val_samples:]], axis = 0)

    train_generator = train_datagen.flow(
        partial_train_data,
        y=partial_train_target,
        batch_size=batch,
        shuffle=True,
    )
    name ='EfficientNetB7_down_0731'
    filepath = base_path +'/model/kfold_confu_matrix/ '+ name + '{val_accuracy:.2f}%-' + str(i) + 'fold.hdf5'
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
    )
    history = final_model.fit_generator(
        #generate_arrays_from_file(trainpath, file_len, file_number, batch_size=batch),
        train_generator,
        #steps_per_epoch=file_number * (file_len / batch),
        epochs=epochs,
        validation_data=(val_data, val_target),
        callbacks=[early_stopping, rlr, checkpoint]
        )
    accu= np.sort(history.history['accuracy'])
    accu_max = np.mean(accu[-2:])

    val_accu = history.history['val_accuracy']
    val_accu_max = np.mean(val_accu[-2:])

    train_acc_list.append(accu_max)
    val_acc_list.append(val_accu_max)
    print('-------fold: ', i)
    print("train Accuracy = ", train_acc_list)
    print("val Accuracy = ", val_acc_list)

    # predict val output
    predict = final_model.predict(val_data)
    predict = np.argmax(predict, axis=1)

    # f1 score
    f1_micro = f1_score(val_target, predict, average='micro')
    print('f1 score: ', f1_micro)

    # plot confusion matrix
    mat = confusion_matrix(val_target, predict, normalize='true')
    print(mat)
    sns.heatmap(mat, square=True, cmap='Blues', annot=True, cbar=False)
    plt.ylabel('true_label')
    plt.xlavel('predicted label')
    plt.title(name)
    plt.savefig(pltsave_path + '/kfold_confu_matrix' + name + 'fold: ' + str(i) + '.png')
    plt.show()
    plt.close('all')


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5)
    plt.title(name)
    plt.savefig(pltsave_path +'/'+ name +'.png') 
    plt.show()

plot_learning_curves(history)
print("train Accuracy = ", train_acc_list)
print("val Accuracy = ", val_acc_list)
print("Average val Accuracy = ", np.mean(val_acc_list))
final_model.save(base_path +'/model/'+ name +'.h5')