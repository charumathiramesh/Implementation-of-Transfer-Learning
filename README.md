# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
![232681405-cc61aa61-1ee8-4303-862d-e90ea1d37f13](https://github.com/charumathiramesh/Implementation-of-Transfer-Learning/assets/120204455/53c107da-3916-4214-9cbd-7278320cc3a8)

## DESIGN STEPS
### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Load CIFAR-10 Dataset & use Image Data Generator to increse the size of dataset

### STEP 3:

Import the VGG-19 as base model & add Dense layers to it


##STEP 4:
Compile and fit the model

##Step 5:
Predict for custom inputs using this model

## PROGRAM
NAME : CHARUMATHI R
REF NO : 212222240021
```python
##Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10
from tensorflow.keras.applications import VGG19
##Load Dataset & Increse the size of it
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
##One Hot Encoding Outputs
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
##Import VGG-19 model & add dense layers
base_model = VGG19(include_top=False, weights = "imagenet",
                   input_shape = (32,32,3))

for layer in base_model.layers:
  layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(250,activation=("relu")))
model.add(Dropout(0.2))
model.add(Dense(100,activation=("relu")))
model.add(Dropout(0.35))
model.add(Dense(10,activation=("softmax")))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

model.fit(x_train, y_train, 
          batch_size=500, epochs=10, 
          validation_data=(x_test, y_test), 
          callbacks=[learning_rate_reduction])
##Metrics
metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(model.predict(x_test), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))
```


```


## OUTPUT
### Training Loss Validation Loss Vs Iteration Plot
 ![download](https://github.com/charumathiramesh/Implementation-of-Transfer-Learning/assets/120204455/3e1a3c52-8039-4d22-89cb-b27f4bda8b84)


### Classification Report
![q](https://github.com/charumathiramesh/Implementation-of-Transfer-Learning/assets/120204455/7a535b96-3042-466e-a089-d8b361d3778a)

### Confusion Matrix
![download](https://github.com/charumathiramesh/Implementation-of-Transfer-Learning/assets/120204455/43ad531a-8276-435e-8962-8e9d951e4005)

## RESULT
Thus, transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture is successfully implemented
