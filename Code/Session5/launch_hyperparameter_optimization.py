#!/bin/env python
import sys
sys.path.append('.')

import session5
import numpy as np
from numpy import random
if __name__ == '__main__':
    
    useServer = True
    useBlock4 = False
    samples_per_epoch = 400
    
    nb_random_trials = 200
    
    #Paremeters to optimize
    batch_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    nb_epochs = [10, 20, 40, 60, 80, 100]
    dropout = [True, False]
    dropout_range = [0, 1] 
    batch_normalization = [True, False]
    optimizers = ['sgd', 'adagrad', 'adadelta']
    learning_rate_range = [0, 1]
    #Parameters for sgd
    momentum_range = [0, 1]
    decay_range = [0, 1]
    nesterov = [True, False]
    
    #Parameters for adagrad
    epsilon_range_adagrad = [0, 1]
    
    #Parameters fro adadelta
    epsilon_range_adadelta = [0, 1]
    rho_range = [0, 1]
    
    
    
    #TODO:
    #This should be random search

#    , bs, samples_per_epoch, nb_epoch
    for i in range(nb_random_trials):
        hyper_parameters = {}
        
        hyper_parameters['batch_size'] = random.choice(batch_sizes)
        hyper_parameters['nb_epoch'] = random.choice(nb_epochs)
        
        hyper_parameters['dropout'] = random.choice(dropout)
        hyper_parameters['dropout_value'] = (dropout_range[1] - dropout_range[0]) * random.random_sample() + dropout_range[0]
        
        hyper_parameters['batch_norm'] = random.choice(batch_normalization)
        hyper_parameters['learning_rate'] = (learning_rate_range[1] - learning_rate_range[0]) * random.random_sample() + learning_rate_range[0]
        
        optimizer = random.choice(optimizers)
        hyper_parameters['optimizer'] = optimizer
                        
        if optimizer == 'sgd':
            
            hyper_parameters['momentum_value'] = (momentum_range[1] - momentum_range[0]) * random.random_sample() + momentum_range[0]
            hyper_parameters['decay_value'] = (decay_range[1] - decay_range[0]) * random.random_sample() + decay_range[0]
            hyper_parameters['nesterov_momentum'] = random.choice(nesterov)
            
        elif optimizer == 'adagrad':
            
            hyper_parameters['epsilon_value'] = (epsilon_range_adagrad[1] - epsilon_range_adagrad[0]) * random.random_sample() + epsilon_range_adagrad[0]
            hyper_parameters['rho_value'] = (rho_range[1] - rho_range[0]) * random.random_sample() + rho_range[0]
        
        else:   
            hyper_parameters['epsilon_value'] = (epsilon_range_adadelta[1] - epsilon_range_adadelta[0]) * random.random_sample() + epsilon_range_adadelta[0]
        
        result, history = session5.launchsession5(useServer, useBlock4, hyper_parameters)







#-Per model
#batch_size = [10, 20, 40, 60, 80, 100]
#epochs = [10, 50, 100]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#learn_rate = [0.0001 0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

#-Per layer:
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform',
#'he_normal', 'he_uniform'] (Not useful in our case)
#-Topology:
#drop-out layers: p % of inactive weights
#batchnormalization
#regularizers

