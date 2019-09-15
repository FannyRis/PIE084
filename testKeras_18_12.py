# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:10:35 2018

@author: louis
"""

import os


currentPath = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(currentPath, '..', 'data_base/')


      
        
        
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#img = load_img(path+'/train/cats/cat.0.jpg')  # this is a PIL image
#x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
## the .flow() command below generates batches of randomly transformed images
## and saves the results to the `preview/` directory
#i = 0
#for batch in datagen.flow(x, batch_size=1,
#                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#    i += 1
#    if i > 20:
#        break  # otherwise the generator would loop indefinitely








from keras.layers.core import Activation, Reshape


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()

input_shape = (100, 100)


#model.add(Reshape(input_shape + (1, ), input_shape=input_shape))

#Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

model.add(Conv2D(filters=2, kernel_size=(3,3), input_shape=(100, 100, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
## the model so far outputs 3D feature maps (height, width, features)
#
#
#
#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
              
model.compile()
              
              
              
              
              
              
              
              
              
              
              
              
batch_size = 7

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data




train_generator = train_datagen.flow_from_directory(
        path+'/train',  # this is the target directory
        target_size=(100, 100),  # all images will be resized to 100x100
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        path+'/test',
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='binary')

imgs, labels = next(train_generator)
imgs = imgs[:,:,:,1]
print(imgs.shape)
print(labels.shape)


#

model.fit_generator(
        train_generator,
        steps_per_epoch=200 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training              