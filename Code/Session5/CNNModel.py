import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
import datetime

def getBaseModel():
    base_model = VGG16(weights='imagenet')
    
    #Do not train base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return base_model

#Option 1: Change 1000 softmax by a 8 softmax
def createModel(dropout_fraction=0.0,batch_normalization=False):
    base_model = getBaseModel()
    
    #Get last but one layer output and add a new layer of 8 predictions
    x = base_model.layers[-2].output
    
    if dropout_fraction > 0.0:
        # Add Dropout layer
        x = Dropout(dropout_fraction)(x)
    
    if batch_normalization:
        # Add Batch normalization layer
        x = BatchNormalization(epsilon=0.001, mode=0, axis=-1,
            momentum=0.99, weights=None, beta_init='zero',
            gamma_init='one', gamma_regularizer=None,
            beta_regularizer=None)(x)

    x = Dense(8, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    
    return model

#Option 2: Use block 4 output
def createModelBlock4():
    base_model = getBaseModel()
    
    #Get last block 4 layer output and add new layers
    x = base_model.layers[-9].output #Output for maxpooling layer of block 4
    
    #Add flatten, fc1, fc2 and softmax after block4
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(8, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)
    
    return model

def compileModel(model,optim):
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
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