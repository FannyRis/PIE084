'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = '../data_base/trainP'
validation_data_dir = '../data_base/validationP'


#nb_train_samples = 2000
#nb_validation_samples = 800
#epochs = 50
#batch_size = 16


nb_train_samples = 200
nb_validation_samples = 80
epochs = 5
batch_size = 16

########################################################
#########------------MODEL PART
########################################################
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
   

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

########################################################
#########------------TRAINING SETTINGS
########################################################
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=360.,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
    
    
########################################################
#########------------VALIDATION PART
########################################################


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

nb_of_batches = len(validation_generator)

########################################################
#########------------TRAINING PART
########################################################
    

#history = model.fit_generator(
#    train_generator,
#    steps_per_epoch=nb_train_samples // batch_size,
#    epochs=epochs,
#    validation_data=validation_generator,
#    validation_steps=nb_validation_samples // batch_size)
#
#model.save_weights('second_try.h5')






#%%
#### TEST AVEC MES MODIFS

model.load_weights('pylons_weights_1st.h5')


#print(model.metrics_names)
#print(model.evaluate_generator(validation_generator))

action = 'predict'
if action == 'validate':
    #valu = [va[1] for va in validation_generator[0:5]]
    #validation_generator[?][0: data, 1: label][nb_in_batch (0-16)]
    true_cat_all = []
    predictions_all = []
    
    for batch_num in range(nb_of_batches):
        predicted_batch = model.predict(validation_generator[batch_num][0])
        for sub_batch in range(len(validation_generator[batch_num][1])):
            true_cat_all.append(validation_generator[batch_num][1][sub_batch])
            predictions_all.append(predicted_batch[sub_batch][0])
    
    
    
    
    
    score_for_pylon = 0.5
    
    predi_bin = [x>0.5 for x in predictions_all]
    print(confusion_matrix(predi_bin, true_cat_all))
    
    
    
    
    
    
    
    for batch_num in range(5):
        predictions = model.predict(validation_generator[batch_num][0])
        #predictions = np.transpose(predictions)
        true_category = validation_generator[batch_num][1]
        true_category = np.transpose(true_category)
        
        ecart = predictions-true_category
        len(ecart)
        
        
        for i in range(batch_size):
            plt.figure()
            plt.imshow(validation_generator[batch_num][0][i])
            plt.title("Predicted: {0} // Truth: {1} // Actual score: {2}".format(
                int(predictions[i]>score_for_pylon), true_category[i], predictions[i][0] ))
            plt.show()

elif action == 'predict':
    from extractImagePartPortion import splitImage
    import os

    currentPath = os.path.dirname(os.path.abspath(__file__))
    imPath = os.path.join(currentPath, '..', 'data', 'gourd_c1818', 'Images', '03')
    imName = '03_00000466.jpg'
    imName = '03_00000654.jpg'
    
    windowSize = 100
    step = 25
    
    [patches, rectanglesX, rectanglesY] = splitImage(imPath, imName, windowSize, step)
    
    patches = np.divide(patches, 255)
    
    
    predicted = model.predict(patches)
    
    print(predicted)
    print(0<predicted.any()<1)
    
   
    score_for_pylon = 0.7
    
    import matplotlib.image as mpimg

    img = mpimg.imread(os.path.join(imPath, imName))
    plt.figure(1)
    plt.imshow(img)
    
#    for ite in [20, 500, 750]:
#        plt.plot(rectanglesX[ite], rectanglesY[ite])
    
    for i in range(len(predicted)):
        if predicted[i] > score_for_pylon:
            plt.plot(rectanglesX[i], rectanglesY[i], 'g')
            
#            plt.figure()
#            plt.imshow(patches[i])
#            plt.title("Predicted: {0} // Actual score: {1}".format(
#                int(predicted[i] > score_for_pylon), predicted[i]))
    plt.show()









#N = 3
#for i in range(N):
#    plt.figure()
#    plt.imshow(validation_generator[1][0][1])
#    plt.imshow(X_test[wrong_predictions,::][idx,::].reshape(28, 28), cmap="Greys")
#    plt.axis("off")
#    
#    
#    plt.title("Predicted: {0} // Truth: {1}".format(
#        Labels[predictions[wrong_predictions][idx]],
#        Labels[y_test[wrong_predictions][idx]]
#    ))




#%%
################### VALIN
#
#import numpy as np
#import matplotlib.pyplot as plt
##%matplotlib inline
#from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#import keras.metrics as metrics
#import random
#
#
#
#plt.figure(figsize=(20,10))
#
#for i, (name, values) in enumerate(history.history.items()):
#    plt.subplot(1, len(history.history.items()), i+1)
#    plt.plot(values)
#    plt.title(name)
#    
#    
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#
#predictions = model.predict(validation_generator).argmax(axis=1)
#
#print("Accuracy: ")
#
#print(accuracy_score(predictions, y_test))
#
#print("\n\n")
#
#print("Confusion matrix: ")
#
#print(confusion_matrix(predictions, y_test))
#
#print("\n\n")
#
#print(classification_report(predictions, y_test, target_names=Labels))
#    
#wrong_predictions = (predictions != y_test)
#
#n_wp = wrong_predictions.sum()
#
#N = 10
#
#for i in range(N):
#    
#    idx = random.choice(range(n_wp))
#    plt.figure()
#    plt.imshow(X_test[wrong_predictions,::][idx,::].reshape(28, 28), cmap="Greys")
#    plt.axis("off")
#    
#    
#    plt.title("Predicted: {0} // Truth: {1}".format(
#        Labels[predictions[wrong_predictions][idx]],
#        Labels[y_test[wrong_predictions][idx]]
#    ))