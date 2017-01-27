#!/bin/env python
import sys
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta

sys.path.append('.')

import CNNData, CNNModel, dataUtils

def launchsession5(useServer, useBlock4, hyper_parameters):
#def launchsession5(useServer, useBlock4, batch_size, samples_per_epoch, nb_epoch):
    
    #Get data
    dataUtils.createDataPaths(useServer,0.7)
    datagen = CNNData.getDataGenerator()
    augmented_datagen = CNNData.getAugmentedDataGenerator()
    train_generator, validation_generator, test_generator = CNNData.getData(datagen, augmented_datagen, hyper_parameters.get('batch_size'))
    
    #Create model
    if useBlock4:
        model = CNNModel.createModelBlock4()
    else:
        model = CNNModel.createModel()
    
    # Usage of optimizers
    if hyper_parameters.get('optimizer') == 'sgd':
        optimizer = SGD(lr=hyper_parameters.get('learning_rate'), decay=hyper_parameters.get('decay_value'), momentum=hyper_parameters.get('momentum_value'), nesterov=hyper_parameters.get('nesterov_momentum'))
    elif hyper_parameters.get('optimizer') == 'adagrad':
        optimizer = Adagrad(lr=hyper_parameters.get('learning_rate'), epsilon=hyper_parameters.get('epsilon_value'), decay=hyper_parameters.get('decay_value'))
    elif hyper_parameters.get('optimizer') == 'adadelta': 
        optimizer = Adadelta(lr=hyper_parameters.get('learning_rate'), rho=hyper_parameters.get('rho_value'), epsilon=hyper_parameters.get('epsilon_value'), decay=hyper_parameters.get('decay_value'))
    
    #Train the model
    model = CNNModel.compileModel(model, optimizer)
    model, history = CNNModel.trainModel(model, train_generator, hyper_parameters, validation_generator)
    
    #Evaluate the model
    CNNModel.plotModelPerformance(history)
    val_result = CNNModel.evaluateModel(model, validation_generator)
    test_result = CNNModel.evaluateModel(model, test_generator)
    
    print 'Validation result ' + str(val_result)
    print 'Test result ' + str(test_result)
    
    return val_result, history



if __name__ == '__main__':
    
    useServer = False
    
    batch_size = 10
    nb_epoch = 20
    samples_per_epoch = 400
    useBlock4 = False
    
    launchsession5(useServer, useBlock4, hyper_parameters)