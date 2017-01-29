# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:29:27 2017

@author: onofre
"""
import cPickle
from os import listdir
from os.path import isfile, join

Hyper_parameters_random_search = cPickle.load( open( "./results/Hyper_parameters_random_search.dat", "rb" ) )


path_params = './Results/Params'
files_params = [f for f in listdir(path_params) if isfile(join(path_params, f))]
path_results = './Results/Results'
files_results = [f for f in listdir(path_results) if isfile(join(path_results, f))]

All_trials_params = []
All_trials_results = []
Adagrad_trials_params = []
Adagrad_trials_results = []
SGD_trials_params = []
SGD_trials_results = []
Adadelta_trials_params = []
Adadelta_trials = []
#Split trials according to its optimizer
for i in range(len(files_params)):
    f = open( files_params[i], "rb" )
    hyper_parameters = cPickle.load(f)
    g = open( files_results[i], "rb" )
    results = cPickle.load(g)
    
    optimizer = hyper_parameters['optimizer']
    All_trials_params.append(hyper_parameters)
    All_trials_results.append(results)                    
    if optimizer == 'sgd':
        
        SGD_trials_params.append(hyper_parameters)
        SGD_trials_results.append(results)
        
    elif optimizer == 'adagrad':
        
        Adagrad_trials_params.append(hyper_parameters)
        Adagrad_trials_results.append(results)
            
    elif optimizer == 'adadelta':  
        Adadelta_trials_params.append(hyper_parameters)
        Adadelta_trials.append(results)
        
    f.close()
    g.close()
    
    
#Find max value in each optimizer    