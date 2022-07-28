import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#from keras.utils import to_categorical

base_path ='/home/irene/deepfashion2/DeepFashion2Dataset'
trainpath = base_path + '/train'
val_path = base_path + '/validation'
pltsave_path = base_path+'/plt_img'

# ------Pattern
#train_data = np.load(os.path.join(trainpath,'inputs1train_pattern2.npy'))
#train_target = np.load(os.path.join(trainpath,'labels1train_pattern2.npy'))

# ------Down
#train_data = np.load(os.path.join(trainpath,'inputs1train_down_0602.npy'))
#train_target = np.load(os.path.join(trainpath,'labels1train_down_0602.npy'))

# ------Sleeve
train_data = np.load(os.path.join(trainpath,'inputs1train_sleeve_0503_clean.npy'))
train_target = np.load(os.path.join(trainpath,'labels1train_sleeve_0503_clean.npy'))

# ------Coat
train_data = np.load(os.path.join(trainpath,'inputs1train_coat_0519_nohand.npy'))
train_target = np.load(os.path.join(trainpath,'labels1train_coat_0519_nohand.npy'))


k = 5
num_val_samples = len(train_data) // k


name ='EfficientNetB7_coat_0803'
kfold_filepath = base_path +'/model/kfold_confu_matrix/ ' 
#model_name = kfold_filepath + 'EfficientNetB7_down_07310.94%-0fold.hdf5'
#model_name = kfold_filepath + 'EfficientNetB7_sleeve_07290.89%-0fold.hdf5'
#model_name = kfold_filepath + 'EfficientNetB7_pattern_07280.98%-0fold.hdf5'
model_name = kfold_filepath + 'EfficientNetB7_coat_07310.93%-0fold.hdf5'

i = 0
val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
val_target = train_target[i * num_val_samples: (i+1) * num_val_samples]
# print(val_target[:10])

# load model and predict
best_model = tf.keras.models.load_model( model_name ) 
predict = best_model.predict(val_data)

predict = np.argmax(predict, axis=1)
val_target = np.argmax(val_target, axis=1)

# f1 score
f1_micro = f1_score(val_target, predict, average='micro')
print(f1_micro)

# plot history confusion
mat = confusion_matrix(val_target, predict, normalize='true')
print(mat)
sns.heatmap(mat, square=True, cmap='Blues', annot=True, cbar=False)
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.savefig(pltsave_path + '/kfold_confu_matrix/' + name + '_ConfusionMatrix_fold' + str(i) + '.png')
plt.show()
plt.close('all')