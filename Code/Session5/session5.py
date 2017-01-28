#!/bin/env python
import sys

sys.path.append('.')

import CNNData, CNNModel, dataUtils, CNNOptimizers
import time

def launchsession5(useServer, useBlock4, batch_size, samples_per_epoch, nb_epoch, optimizer, dropout_fraction=0.0, batch_normalization=False, random_search = False):
    
    start = time.time()
    #Get data
    dataUtils.createDataPaths(useServer,0.7)
    datagen = CNNData.getDataGenerator()
    augmented_datagen = CNNData.getAugmentedDataGenerator()
    train_generator, validation_generator, test_generator = CNNData.getData(datagen, augmented_datagen, batch_size)
    
    #Create model
    if useBlock4:
        model = CNNModel.createModelBlock4()
    else:
        model = CNNModel.createModel(dropout_fraction=dropout_fraction,batch_normalization=batch_normalization)
    
    #Train the model
    model = CNNModel.compileModel(model, optimizer)
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
    
    batch_size = 10
    nb_epoch = 20
    samples_per_epoch = 400
    useBlock4 = False
    
    learning_rate=0.01
    
    optimizer=CNNOptimizers.getOptimizer('adagrad',learning_rate,rho_value=0.95,decay_value=0.0,epsilon_value=1e-08,momentum_value=0.0,nesterov_momentum=False)
    
    launchsession5(useServer, useBlock4, batch_size, samples_per_epoch, nb_epoch, optimizer)