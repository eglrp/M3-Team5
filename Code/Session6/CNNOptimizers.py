from keras.optimizers import SGD, Adagrad, Adadelta

def getOptimizer(optimizerChoice, learning_rate, rho_value = 0.95, decay_value = 0.0, epsilon_value = 1e-08, momentum_value = 0.0, nesterov_momentum = False):
    if optimizerChoice == 'sgd':
        optimizer = SGD(lr = learning_rate, decay = decay_value, momentum = momentum_value, nesterov = nesterov_momentum)
    elif optimizerChoice == 'adagrad':
        optimizer = Adagrad(lr = learning_rate, epsilon = epsilon_value, decay = decay_value)
    elif optimizerChoice == 'adadelta': 
        optimizer = Adadelta(lr = learning_rate, rho = rho_value, epsilon = epsilon_value, decay = decay_value)
    
    return optimizer
