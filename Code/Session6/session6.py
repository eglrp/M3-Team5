#!/bin/env python
import sys

sys.path.append('.')

import CNNData, CNNModel, dataUtils, CNNOptimizers
import time

def launchsession6(useServer, batch_size, samples_per_epoch, nb_epoch, optimizer_type, dropout_fraction=0.0, batch_normalization=False, random_search = False):
    
    start = time.time()
    #Get data
    dataUtils.createDataPaths(useServer, 0.7)
    datagen = CNNData.getDataGenerator()
    augmented_datagen = CNNData.getAugmentedDataGenerator()
    train_generator, validation_generator, test_generator = CNNData.getDataOld(datagen, augmented_datagen, batch_size)
    
    #Create model
    model = CNNModel.createModel(dropout_fraction = dropout_fraction, batch_normalization = batch_normalization)
    
    #Train the model
    model = CNNModel.compileModel(model, optimizer_type)
    model.summary()
    model, history = CNNModel.trainModel(model, train_generator, samples_per_epoch, nb_epoch, validation_generator)
    
    #Evaluate the model
    if random_search:
        CNNModel.plotModelPerformanceRandom(history)
    else:    
        CNNModel.plotModelPerformance(history)
    val_result = CNNModel.evaluateModel(model, validation_generator)
    test_result = CNNModel.evaluateModel(model, test_generator)
    
    print 'Validation result ' + str(val_result)
    print 'Test result ' + str(test_result)
    
    end = time.time()
    time_expend = end - start
    print 'Done in ' + str(time_expend) + ' secs.'
    
    
    return test_result, val_result, time_expend, history


if __name__ == '__main__':
    
    useServer = True
    
    batch_size = 32
    nb_epoch = 15
    samples_per_epoch = 1000
    dropout_fraction = 0.0
    
    learning_rate = 1
    
    optimizer_type = CNNOptimizers.getOptimizer('adadelta', learning_rate, 
                                           rho_value = 0.95, decay_value = 0.0,
                                           epsilon_value = 1e-08,
                                           momentum_value = 0.0,
                                           nesterov_momentum = False)
    
    launchsession6(useServer, batch_size, samples_per_epoch, nb_epoch, optimizer_type, dropout_fraction)
