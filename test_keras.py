# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:33:42 2018

@author: louis
"""
#from keras.preprocessing import image
#
#import matplotlib.pyplot as plt
#import numpy as np
#img = np.random.rand(224,224,3)
#plt.imshow(img)
#plt.show()
#
#img_path = "image.jpeg"
#img = image.load_img(img_path, target_size=(224, 224))
#print(type(img))
#
#x = image.img_to_array(img)
#print(type(x))
#print(x.shape)
#plt.imshow(x)





#import os
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
#datagen = ImageDataGenerator(
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')
#
#currentPath = os.path.dirname(os.path.abspath(__file__))
#
#imPath = os.path.join(currentPath, '..', 'data_base', 'pylonsTrain', '03_00000178_pyl_198_2301_1108.jpg')
#
#img = load_img(imPath)  # this is a PIL image
#x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
## the .flow() command below generates batches of randomly transformed images
## and saves the results to the `preview/` directory
#i = 0
#for batch in datagen.flow(x, batch_size=1,
#                          save_to_dir='../preview', save_prefix='cat', save_format='jpeg'):
#    i += 1
#    if i > 20:
#        break  # otherwise the generator would loop indefinitely
#
#

#%%


import os
import keras
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.metrics as metrics
import random



def top3_acc(ytrue, ypred):
    return metrics.top_k_categorical_accuracy(ytrue, ypred, k=2)

currentPath = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(currentPath, '..', 'data_base/')



from keras.preprocessing import image 

batch_size = 4
gen=image.ImageDataGenerator()

trn_batches = gen.flow_from_directory(
        path+'train',
        target_size=(100,100),
        class_mode='categorical')

val_batches = gen.flow_from_directory(
        path+'test',
        target_size=(100,100),shuffle=False,
        batch_size=batch_size,
        class_mode='categorical')
        

imgs, labels = next(trn_batches)        
print(imgs[0])
print(labels[2])



# MODEL PART ------------

model = Sequential()

model.add(Dense(units=10, input_shape=(100,100,3)))  #couche dense a 100 neuronnes, sur le jeu d'apprentissage
model.add(Activation('relu')) #fonction sigmoid


model.add(Dense(units=10)) #couche dense a 10 neuronnes, sur le jeu d'apprentissage
model.add(Activation('softmax')) #prend toutes les valeurs des neuronnes, passe a l'exp, moyenne



# Change decay for better results

# lr: 1e-3, decay: 0
n_batch = 3

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001, decay=0.5/n_batch, nesterov=False),
            metrics=['accuracy'])


history = model.fit(trn_batches, epochs=5, batch_size=2) #epoch : nb de passages, 
    #batch_size : nb d'exemples montres en meme temps, permet de moins overfitter
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
plt.figure(figsize=(20,10))

for i, (name, values) in enumerate(history.history.items()):
    plt.subplot(1, len(history.history.items()), i+1)
    plt.plot(values)
    plt.title(name)
    
    
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

predictions = model.predict(X_test).argmax(axis=1)

print("Accuracy: ")

print(accuracy_score(predictions, y_test))

print("\n\n")

print("Confusion matrix: ")

print(confusion_matrix(predictions, y_test))

print("\n\n")

print(classification_report(predictions, y_test, target_names=Labels))
    
wrong_predictions = (predictions != y_test)

n_wp = wrong_predictions.sum()

N = 10

for i in range(N):
    
    idx = random.choice(range(n_wp))
    plt.figure()
    plt.imshow(X_test[wrong_predictions,::][idx,::].reshape(28, 28), cmap="Greys")
    plt.axis("off")
    
    
    plt.title("Predicted: {0} // Truth: {1}".format(
        Labels[predictions[wrong_predictions][idx]],
        Labels[y_test[wrong_predictions][idx]]
    ))



