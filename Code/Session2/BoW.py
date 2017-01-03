import time
import cPickle
from sklearn import cluster
import numpy as np
import spatial_pyramid as spt_py

def computeCodebook(D,k):
    print 'Computing kmeans with '+str(k)+' centroids'
    init=time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4)
    codebook.fit(D)
    cPickle.dump(codebook, open("codebook.dat", "wb"))
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'

    return codebook
    
def getVisualWords(codebook,k,Train_descriptors):
    print 'Computing visual words'
    init=time.time()
    visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        words=codebook.predict(Train_descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k)
    
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return visual_words
    
def getVisualWordsSpatialPyramid(codebook, k, Train_descriptors, Train_image_size, Train_keypoints):
    print 'Computing visual words'
    init=time.time()
    visual_words=np.zeros((len(Train_descriptors),21*k),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        visual_words[i, :] = spt_py.spatial_pyramid(Train_image_size[i], Train_descriptors[i], Train_keypoints[i], codebook, k)
    
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return visual_words    