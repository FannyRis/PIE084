# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:02:18 2018

@author: Louis BAETENS
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from utils import mnist_reader
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.metrics as metrics
import random


def top3_acc(ytrue, ypred):
    return metrics.top_k_categorical_accuracy(ytrue, ypred, k=3)


def main():
    Labels = ["0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"]
    
    X_train, y_train = mnist_reader.load_mnist('data/mnist', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/mnist', kind='t10k')
    
    N = X_train.shape[0]
    batch_size = 32
    
    n_batch = int(N/batch_size)
    
    y_train_ohe = to_categorical(y_train)
    
    

    
    model = Sequential()
    
    model.add(Dense(units=50, input_dim=X_train.shape[1])) #couche dense a 100 neuronnes, sur le jeu d'apprentissage
    model.add(Activation('relu')) #fonction sigmoid
    
    
    model.add(Dense(units=10)) #couche dense a 10 neuronnes, sur le jeu d'apprentissage
    model.add(Activation('softmax')) #prend toutes les valeurs des neuronnes, passe a l'exp, moyenne
    

    
    # Change decay for better results
    
    # lr: 1e-3, decay: 0
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, decay=0.5/n_batch, nesterov=False),
                 metrics=[metrics.categorical_accuracy, top3_acc])



    history = model.fit(X_train, y_train_ohe, epochs=10, batch_size=32) #epoch : nb de passages, 
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


















#def main():
#    # 3. Import libraries and modules
#    np.random.seed(123)  # for reproducibility
#         
#    # 4. Load pre-shuffled MNIST data into train and test sets
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     
#    # 5. Preprocess input data
#    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#    X_train /= 255
#    X_test /= 255
#     
#    # 6. Preprocess class labels
#    Y_train = np_utils.to_categorical(y_train, 10)
#    Y_test = np_utils.to_categorical(y_test, 10)
#     
#    # 7. Define model architecture
#    model = Sequential()
#     
#    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
#    model.add(Convolution2D(32, 3, 3, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#    model.add(Dropout(0.25))
#     
#    model.add(Flatten())
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(10, activation='softmax'))
#     
#    # 8. Compile model
#    model.compile(loss='categorical_crossentropy',
#                  optimizer='adam',
#                  metrics=['accuracy'])
#     
#    # 9. Fit model on training data
#    model.fit(X_train, Y_train, 
#              batch_size=32, nb_epoch=10, verbose=1)
#     
#    # 10. Evaluate model on test data
#    score = model.evaluate(X_test, Y_test, verbose=0)

if __name__ == '__main__':
    main()