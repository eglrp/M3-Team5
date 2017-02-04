import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l1, l1l2, l1
import datetime

 
def createModel(dropout_fraction = 0.0, batch_normalization = False):
    
    input = Input(shape = (3, 256, 256))
    
    x = Convolution2D(32, 5, 5, activation = 'relu', W_regularizer=l2(0.01), border_mode = 'same', name = 'conv1')(input)
    
    #x = Dropout(dropout_fraction)(x)
    
    x = MaxPooling2D((4, 4), strides = (4, 4), name = 'pool1')(x)

    x = BatchNormalization()(x)

    x = Convolution2D(32, 5, 5, activation = 'relu', W_regularizer=l2(0.01), border_mode = 'same', name = 'conv2')(x)
    
    #x = Dropout(dropout_fraction)(x)
    
    x = MaxPooling2D((4, 4), strides = (4, 4), name = 'pool2')(x)

    x = BatchNormalization()(x)
    
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'pool3')(x)

    #Classification block

    x = Flatten(name = 'flatten')(x)
    
   #x = Dropout(dropout_fraction)(x)

    x = Dense(512, activation = 'relu', name = 'fc1')(x)
    
   #x = Dropout(dropout_fraction)(x)
    
    x = Dense(8, activation = 'softmax', name = 'predictions')(x)

    model = Model(input = input, output = x)
    
    return model

def createModel02(dropout_fraction = 0.0, batch_normalization = False):
    
    img_input = Input(shape = (3, 256, 256))
    x = img_input
    #First convolutional layer
    x = Convolution2D(32, 3, 3, 
                  init = 'glorot_uniform', 
                  activation = None, 
                  border_mode = 'valid', #'valid', 'same' or 'full'. ('full' requires the Theano backend.)
                  subsample = (1, 1), 
                  dim_ordering = 'default', 
                  W_regularizer = None, 
                  b_regularizer = None, 
                  activity_regularizer = None, 
                  W_constraint = None, 
                  b_constraint = None, 
                  bias = True, name = 'conv01')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D(pool_size = (2, 2), 
                     strides = (3, 3), 
                     border_mode = 'valid', 
                     dim_ordering = 'default',
                     name = 'pool01')(x) 
    
    x = Convolution2D(64, 3, 3, 
                  init = 'glorot_uniform', 
                  activation = None, 
                  border_mode = 'valid', #'valid', 'same' or 'full'. ('full' requires the Theano backend.)
                  subsample = (1, 1), 
                  dim_ordering = 'default', 
                  W_regularizer = None, 
                  b_regularizer = None, 
                  activity_regularizer = None, 
                  W_constraint = None, 
                  b_constraint = None, 
                  bias = True, name = 'conv02')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2, 2), 
                     strides = (3, 3), 
                     border_mode = 'valid', 
                     dim_ordering = 'default',
                     name = 'pool02')(x) 
    x = Convolution2D(128, 3, 3, 
                  init = 'glorot_uniform', 
                  activation = None, 
                  border_mode = 'valid', #'valid', 'same' or 'full'. ('full' requires the Theano backend.)
                  subsample = (1, 1), 
                  dim_ordering = 'default', 
                  W_regularizer = None, 
                  b_regularizer = None, 
                  activity_regularizer = None, 
                  W_constraint = None, 
                  b_constraint = None, 
                  bias = True, name = 'conv03')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2, 2), 
                     strides = (3, 3), 
                     border_mode = 'valid', 
                     dim_ordering = 'default',
                     name = 'pool03')(x)
    if dropout_fraction > 0.0:
        # Add Dropout layer
        x = Dropout(dropout_fraction)(x)
    
    if batch_normalization:
        # Add Batch normalization layer
        x = BatchNormalization(epsilon=0.001, mode=0, axis=-1,
            momentum=0.99, weights=None, beta_init='zero',
            gamma_init='one', gamma_regularizer=None,
            beta_regularizer=None)(x)
        
        
#    x = MaxPooling2D(pool_size = (2, 2), 
#                     strides = None, 
#                     border_mode = 'valid', 
#                     dim_ordering = 'default')(x)  
    #Flatten the results to put inside a dense layer    
    x = Flatten(name = 'flatten')(x)
    #Dense layer to make the classification
    x = Dense(200, init = 'glorot_uniform', 
              activation = None, 
              weights = None,
              W_regularizer = None, 
              b_regularizer = None, 
              activity_regularizer = None, 
              W_constraint = None, 
              b_constraint = None, 
              bias = True, 
              input_dim = None)(x)
    x = Activation('relu')(x)
    x = Dense(8, activation = 'softmax', name = 'predictions')(x)
    model = Model(input = img_input, output = x)
    print model.summary()
    
    return model

def compileModel(model,optimizer):
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


def trainModel(model, train_generator, samples_per_epoch, nb_epoch, validation_generator):
    history=model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,nb_epoch=nb_epoch,validation_data=validation_generator,nb_val_samples=800)
    
    return model, history

def plotModelPerformance(history):
    plt.figure(1)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.legend(loc=3)
    plt.savefig('accHistory.eps', format='eps', dpi=1000)
    
    plt.figure(2)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend(loc=3)
    plt.savefig('lossHistory.eps', format='eps', dpi=1000)

def plotModelPerformanceRandom(history):
    plt.switch_backend('PS')
    current_time = datetime.datetime.now().strftime("%d,%Y,%I%M%p")
    plt.figure(1)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.legend(loc=3)
    plt.savefig('./Results/History/accHistory'+ current_time +'.eps', format='eps', dpi=1000)

    plt.figure(2)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend(loc=3)
    plt.savefig('./Results/History/lossHistory'+ current_time +'.eps', format='eps', dpi=1000)
    
def evaluateModel(model, test_generator):
    result = model.evaluate_generator(test_generator, val_samples=800)
    
    return result

def createModel_template(dropout_fraction = 0.0, batch_normalization = False):
    
    img_input = Input(shape = (3, 256, 256))
    x = img_input
    #First convolutional layer
    x = Convolution2D(nb_filter, nb_row, nb_col, 
                  init = 'glorot_uniform', 
                  activation = None, 
                  border_mode = 'valid', #'valid', 'same' or 'full'. ('full' requires the Theano backend.)
                  subsample = (1, 1), 
                  dim_ordering = 'default', 
                  W_regularizer = None, 
                  b_regularizer = None, 
                  activity_regularizer = None, 
                  W_constraint = None, 
                  b_constraint = None, 
                  bias = True, name = 'conv01')(x)
    x = Activation()(x)
    
    if dropout_fraction > 0.0:
        # Add Dropout layer
        x = Dropout(dropout_fraction)(x)
    
    if batch_normalization:
        # Add Batch normalization layer
        x = BatchNormalization(epsilon=0.001, mode=0, axis=-1,
            momentum=0.99, weights=None, beta_init='zero',
            gamma_init='one', gamma_regularizer=None,
            beta_regularizer=None)(x)
        
        
    x = MaxPooling2D(pool_size = (2, 2), 
                     strides = None, 
                     border_mode = 'valid', 
                     dim_ordering = 'default')(x)  
    #Flatten the results to put inside a dense layer    
    x = Flatten(name = 'flatten')(x)
    #Dense layer to make the classification
    x = Dense(output_dim, init = 'glorot_uniform', 
              activation = None, 
              weights = None,
              W_regularizer = None, 
              b_regularizer = None, 
              activity_regularizer = None, 
              W_constraint = None, 
              b_constraint = None, 
              bias = True, 
              input_dim = None)(x)
    x = Activation()(x)
    x = Dense(8, activation = 'softmax', name = 'predictions')(x)
    model = Model(input = img_input, output = x)
    
    return model
