import time
from yael import ynumpy
import numpy as np
import spatial_pyramid as spt_py

def getGMM(D,k):
    print 'Computing gmm with '+str(k)+' centroids'
    init=time.time()
    gmm = ynumpy.gmm_learn(np.float32(D), k)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return gmm

def getFisherVectors(Train_descriptors,k,gmm):
    print 'Computing Fisher vectors'
    d = int(Train_descriptors[0].shape[1])
    init=time.time()
    fisher=np.zeros((len(Train_descriptors),k*d*2),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        fisher[i,:]= ynumpy.fisher(gmm, Train_descriptors[i], include = ['mu','sigma'])
    
    
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    return fisher
    
    
def getFisherVectorsSpatialPyramid(Train_descriptors, k, gmm, Train_image_size, Train_keypoints, Use_spatial_pyramid):
    print 'Computing Fisher vectors'
    num_subim = list(np.append([1], [Use_spatial_pyramid[i][0]*Use_spatial_pyramid[i][1] for i in range(len(Use_spatial_pyramid))]))
    num_grids = sum(num_subim)
    d = int(Train_descriptors[0].shape[1])
    init = time.time()
    fisher = np.zeros((len(Train_descriptors), (k*d*2)*num_grids), dtype=np.float32)
    
    for i in xrange(len(Train_descriptors)):
        fisher[i,:] = spt_py.spatial_pyramid_fisher(Train_image_size[i], Train_descriptors[i], Train_keypoints[i], k, gmm, Use_spatial_pyramid)
        
    end = time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    return fisher       