import time
from yael import ynumpy
import numpy as np

def getGMM(D,k):
    print 'Computing gmm with '+str(k)+' centroids'
    init=time.time()
    gmm = ynumpy.gmm_learn(np.float32(D), k)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return gmm

def getFisherVectors(Train_descriptors,k,gmm):
    init=time.time()
    fisher=np.zeros((len(Train_descriptors),k*128*2),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        fisher[i,:]= ynumpy.fisher(gmm, Train_descriptors[i], include = ['mu','sigma'])
    
    
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    return fisher