# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:00:51 2018

@author: khanhle
"""

import numpy
# fix random seed for reproducibility

seed = 7
numpy.random.seed(seed)
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras import optimizers
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
import math
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback

print(__doc__)

import matplotlib.pyplot as plt


#define params
trn_file = 'dataset/pssm.cv.csv'
tst_file = 'dataset/pssm.ind.csv'

nb_classes = 2
nb_kernels = 3
nb_pools = 2
epochs = 150
batch_size = 10

# load training dataset
dataset = numpy.loadtxt(trn_file, delimiter=",", ndmin = 2)
# split into input (X) and output (Y) variables
X = dataset[:,0:400].reshape(len(dataset),1,20,20)
Y = dataset[:,400]

Y = np_utils.to_categorical(Y,nb_classes)
#print X,Y
#nb_classes = Y.shape[1]
#print nb_classes

# load testing dataset
dataset1 = numpy.loadtxt(tst_file, delimiter=",", ndmin = 2)
# split into input (X) and output (Y) variables
X1 = dataset1[:,0:400].reshape(len(dataset1),1,20,20)
Y1 = dataset1[:,400]
true_labels = numpy.asarray(Y1)

Y1 = np_utils.to_categorical(Y1,nb_classes)
# i = 3571
# plt.imshow(X[i,0], interpolation='nearest')
# print('label:', Y[i,:])

def cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape = (1,20,20)))
    model.add(Conv2D(32, (nb_kernels, nb_kernels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (nb_kernels, nb_kernels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (nb_kernels, nb_kernels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), data_format="channels_first"))

    ## add the model on top of the convolutional base
    model.add(Flatten())
#    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.1))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Compile model
    return model

def plot_fig(i, history):
    fig = plt.figure()
    plt.plot(range(1,epochs+1),history.history['val_acc'],label='validation')
    plt.plot(range(1,epochs+1),history.history['acc'],label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1,epochs])
#     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig('img/'+str(i)+'-accuracy.png')
    plt.close(fig)

#model1 = cnn_model()
#print(model1.summary())

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

#tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=50, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

#RMSProp
#model_rmsprop = model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
#hist_rmsprop = model.fit(X, Y, nb_epoch=200, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1))
#
#model = cnn_model()
##Adam
#model_adam = model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
#hist_adam = model.fit(X, Y, nb_epoch=200, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1))
#
#model = cnn_model()
##Nadam
#model_nadam = model.compile(loss='categorical_crossentropy', optimizer="nadam", metrics=['accuracy'])
#hist_nadam = model.fit(X, Y, nb_epoch=200, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1))

model = cnn_model()
#SGD + constant learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
hist_sgd_constant = model.fit(X, Y, nb_epoch=epochs, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1))

# plot model accuracy
#plot_fig(1, hist_sgd_constant)

#model = cnn_model()
#SGD + Time-Based Decay

model1 = cnn_model()
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
model1.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False), metrics=['accuracy'])
hist_sgd_decay = model1.fit(X, Y, nb_epoch=epochs, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1))

# plot model accuracy
#plot_fig(2, hist_sgd_decay)

# SGD + Step Decay
model = cnn_model()

# define SGD optimizer
momentum = 0.5

model_sgd_step = model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False), metrics=['accuracy'])

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))
       print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

# learning schedule callback
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]

lrate = LearningRateScheduler(step_decay)

# fit the model
hist_sgd_step = model.fit(X, Y, 
                     validation_data=(X1, Y1), 
                     epochs=epochs, 
                     batch_size=10, 
                     callbacks=callbacks_list, 
                     verbose=2)
       
# plot model accuracy
#plot_fig(3, hist_sgd_step)

# SGD + exponental decay
model = cnn_model()

# define SGD optimizer
momentum = 0.8

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False), metrics=['accuracy'])

# define step decay function
class LossHistory_(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('lr:', exp_decay(len(self.losses)))

def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * numpy.exp(-k*epoch)
    return lrate

# learning schedule callback
loss_history_ = LossHistory_()
lrate_ = LearningRateScheduler(exp_decay)
callbacks_list_ = [loss_history_, lrate_]

# fit the model
hist_sgd_expo = model.fit(X, Y, 
     validation_data=(X1, Y1), 
     epochs=epochs, 
     batch_size=10, 
     callbacks=callbacks_list_, 
     verbose=2)

# plot model accuracy
plot_fig(4, hist_sgd_expo)


# fit CNN model using Adagrad optimizer
model = cnn_model()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Nadam(),
              metrics=['accuracy'])
hist_adagrad = model.fit(X, Y, 
                     validation_data=(X1, Y1), 
                     epochs=epochs, 
                     batch_size=batch_size,
                     verbose=2)

# fit CNN model using Adadelta optimizer
model = cnn_model()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adadelta(),
              metrics=['accuracy'])
hist_adadelta = model.fit(X, Y, 
                     validation_data=(X1, Y1), 
                     epochs=epochs, 
                     batch_size=batch_size,
                     verbose=2)

# fit CNN model using RMSprop optimizer
model = cnn_model()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])
hist_rmsprop = model.fit(X, Y, 
                     validation_data=(X1, Y1), 
                     epochs=epochs, 
                     batch_size=batch_size,
                     verbose=2)

# fit CNN model using Adam optimizer
model8 = cnn_model()
model8.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
hist_adam = model8.fit(X, Y, 
                     validation_data=(X1, Y1), 
                     epochs=epochs, 
                     batch_size=batch_size,
                     verbose=2)

fig = plt.figure(figsize=(20,12))
#plt.plot(range(epochs),hist_sgd_constant.history['val_acc'],label='SGD Constant lr')
#plt.plot(range(epochs),hist_sgd_decay.history['val_acc'],label='SGD Time-based')
plt.plot(range(epochs),hist_sgd_step.history['val_acc'],label='SGD')
#plt.plot(range(epochs),hist_sgd_expo.history['val_acc'],label='SGD Exponential decay')
plt.plot(range(epochs),hist_adagrad.history['val_acc'],label='Nadam')
plt.plot(range(epochs),hist_adadelta.history['val_acc'],label='Adadelta')
plt.plot(range(epochs),hist_rmsprop.history['val_acc'],label='RMSprop')
plt.plot(range(epochs),hist_adam.history['val_acc'],label='Adam')
plt.legend(loc=0, prop={'size': 20})
plt.xlabel('epochs', fontsize=20)
plt.xlim([0,epochs])
plt.ylabel('validation accuracy', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.title("Comparing Model Accuracy", fontsize=20)
plt.show()
fig.savefig('img/compare-accuracy.png')
plt.close(fig)