#!/bin/env python
import sys
sys.path.append('.')

import CNNData, CNNModel, dataUtils

def launchsession5(useServer, useBlock4, batch_size, samples_per_epoch, nb_epoch):
    
    #Get data
    dataUtils.createDataPaths(useServer,0.7)
    datagen = CNNData.getDataGenerator()
    augmented_datagen = CNNData.getAugmentedDataGenerator()
    train_generator, validation_generator, test_generator = CNNData.getData(datagen, augmented_datagen, batch_size)
    
    #Create model
    if useBlock4:
        model = CNNModel.createModelBlock4()
    else:
        model = CNNModel.createModel()
    
    #Train the model
    model = CNNModel.compileModel(model, optim='adadelta')
    model, history = CNNModel.trainModel(model, train_generator, batch_size, samples_per_epoch, nb_epoch, validation_generator)
    
    #Evaluate the model
    CNNModel.plotModelPerformance(history)
    val_result = CNNModel.evaluateModel(model, validation_generator)
    test_result = CNNModel.evaluateModel(model, test_generator)
    
    print 'Validation result ' + str(val_result)
    print 'Test result ' + str(test_result)
    
    return val_result


if __name__ == '__main__':
    
    useServer = True
    
    batch_size = 10
    nb_epoch = 20
    samples_per_epoch = 400
    useBlock4 = False
    
    launchsession5(useServer, useBlock4, batch_size, samples_per_epoch, nb_epoch)
