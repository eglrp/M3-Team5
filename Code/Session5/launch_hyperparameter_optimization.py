#!/bin/env python
import sys
sys.path.append('.')

import session5

if __name__ == '__main__':
    
    useServer = True
    useBlock4 = False
    
    batch_sizes = [10, 20, 40, 60, 80, 100]
    nb_epoch = [10, 50, 100]
    samples_per_epoch = 400
    
    #TODO:
    #This should be random search
    
    for bs in batch_sizes:
        for epoch in nb_epoch:
            print 'For batch size '+ bs+' , epoch ' + epoch
            result = session5.launchsession5(useServer, useBlock4, bs, samples_per_epoch, nb_epoch)
