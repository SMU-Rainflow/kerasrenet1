from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import resnet
import pickle



early_stopping=EarlyStopping(monitor='val_loss', min_delta=0.02,
                              patience=3, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

data = load_obj("../pkl/new1-64-length")
y = [int(i)-1 for i in data[2]]
x = data[0] 

x = np.array(x)
y = np.eye(24)[y]

batch_size = 32
nb_classes = 24
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 64, 64
# The CIFAR10 images are RGB.
img_channels = 3

model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x,y,validation_split=0.3,callbacks = [early_stopping],epochs=50)

model.save("test6450.h5")
save_obj(history,"resnet_his")