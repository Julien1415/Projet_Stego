import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Input, \
                        Dense, Flatten, BatchNormalization, Reshape, \
                        GlobalAveragePooling1D, Dropout

import numpy as np
from keras.utils import np_utils

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import signal
from pylab import *

from skimage.util import img_as_float
import time
import datetime
import os
os.chdir("C:/Users/Julie/OneDrive/Documents/Stage_M2/Ressources/Codes_tableaux")    #chemin vers le dossier de travail

from fingerprinting import *


## Récupération des données
BD_Cover = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Alaska_22759'
BD_finger = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/Fingerprinted_seuil0.3'
BD_JMiPOD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/JMiPOD_2063'
BD_JUNIWARD = 'C:/Users/Julie/OneDrive/Documents/Stage_M2/Travail/JUNIWARD_6651'

list_files_Cover = [files for files in os.listdir(BD_Cover) if os.path.isfile(os.path.join(BD_Cover,files))]
list_files_finger = [files for files in os.listdir(BD_finger) if os.path.isfile(os.path.join(BD_finger,files))]
list_files_JMiPOD = [files for files in os.listdir(BD_JMiPOD) if os.path.isfile(os.path.join(BD_JMiPOD,files))]
list_files_JUNIWARD = [files for files in os.listdir(BD_JUNIWARD) if os.path.isfile(os.path.join(BD_JUNIWARD,files))]

Nbre_images = 500


X_finger = []
label_finger = np.zeros((Nbre_images*2,2))
for k in range(Nbre_images):
    img_path = os.path.join(BD_Cover,list_files_Cover[k])
    img = mpimg.imread(img_path)
    img = img_as_float(img)
    X_finger.append(img)
    label_finger[k,0]=1
for k in range(Nbre_images):
    img_path = os.path.join(BD_finger,list_files_finger[k])
    img = mpimg.imread(img_path)
    img = img_as_float(img)
    X_finger.append(img)
    label_finger[Nbre_images+k,1]=1
X_finger = np.array(X_finger)


X_JMiPOD = []
label_JMiPOD = np.zeros((Nbre_images*2,2))
for k in range(Nbre_images):
    img_path = os.path.join(BD_Cover,list_files_Cover[k])
    img = mpimg.imread(img_path)
    img = img_as_float(img)
    X_JMiPOD.append(img)
    label_JMiPOD[k,0]=1
for k in range(Nbre_images):
    img_path = os.path.join(BD_JMiPOD,list_files_JMiPOD[k])
    img = mpimg.imread(img_path)
    img = img_as_float(img)
    X_JMiPOD.append(img)
    label_JMiPOD[Nbre_images+k,1]=1
X_JMiPOD = np.array(X_JMiPOD)


X_JUNIWARD = []
label_JUNIWARD = np.zeros((Nbre_images*2,2))
for k in range(Nbre_images):
    img_path = os.path.join(BD_Cover,list_files_Cover[k])
    img = mpimg.imread(img_path)
    img = img_as_float(img)
    X_JUNIWARD.append(img)
    label_JUNIWARD[k,0]=1
for k in range(Nbre_images):
    img_path = os.path.join(BD_JUNIWARD,list_files_JUNIWARD[k])
    img = mpimg.imread(img_path)
    img = img_as_float(img)
    X_JUNIWARD.append(img)
    label_JUNIWARD[Nbre_images+k,1]=1
X_JUNIWARD = np.array(X_JUNIWARD)

# X_merged = np.concatenate((X_JMiPOD,X_JUNIWARD), axis=0)
# label_merged = np.concatenate((label_JMiPOD,label_JUNIWARD), axis=0)

## Split data
start_global_time = time.time()

X_train, X_test , y_train , y_test = train_test_split(X_JMiPOD,label_JMiPOD,test_size=0.2,stratify=label_JMiPOD)
X_train, X_test , y_train , y_test = train_test_split(X_finger,label_finger,test_size=0.2,stratify=label_finger)
# X_train0, X_test0 , y_train0 , y_test0 = train_test_split(X_JUNIWARD,label_JUNIWARD,test_size=0.2,stratify=label_JUNIWARD)
# X_train, X_test , y_train , y_test = train_test_split(X_JUNIWARD,label_JUNIWARD,test_size=0.2,stratify=lael_JUNIWARD)
# X_train, X_test , y_train , y_test = train_test_split(X_merged,label_merged,test_size=0.2,stratify=label_merged)

delta_global = round(time.time() - start_global_time)
print("Temps pour split ",datetime.timedelta(seconds=delta_global))

## Création du CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=8, kernel_size=(5,5),activation="relu",padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters=5, kernel_size=(3,3),activation="relu",padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(2,activation="softmax"),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Training et tests, courbes de précision du CNN
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("loss : ", test_loss)
print("accuracy : ", test_acc)
rep_rnn = model.predict(X_test)

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()



### CNN2

## Création du CNN2
model2 = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(5,5),activation="relu",padding="same",input_shape=(512,512,3)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(filters=16, kernel_size=(3,3),activation="relu",padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(2,activation="softmax"),
])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Training et tests, courbes de précision du CNN2
history = model2.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=1)

test_loss, test_acc = model2.evaluate(X_test, y_test)
print("loss : ", test_loss)
print("accuracy : ", test_acc)
rep_rnn = model2.predict(X_test)

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()