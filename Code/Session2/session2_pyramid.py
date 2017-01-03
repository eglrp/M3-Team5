#!/bin/env python
import sys
sys.path.append('.')

import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
import spatial_pyramid
#import descriptors, SVMClassifiers, Evaluation, inputOutputUtils, PCA_computing

def launchsession2_pyramid():
    start = time.time()
    
    # read the train and test files
    
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))
    
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    # create the SIFT detector object
    
    SIFTdetector = cv2.SIFT(nfeatures=100)
    
    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    
    Train_descriptors = []
    Train_label_per_descriptor = []
    #Is a list where each position conatains a list of all keypoints for a certain image
    Train_keypoints = []
    Train_image_size = []
    for i in range(len(train_images_filenames)):
        filename = train_images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        Train_image_size.append(np.float32(gray.shape))
        Train_keypoints.append(kpt)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print str(len(kpt)) + ' extracted keypoints and descriptors'
    
    # Transform everything to numpy arrays
    size_descriptors = Train_descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype = np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
        startingpoint += len(Train_descriptors[i])
    
    #Computing bag of words using k-means
    k = 1024
    
    print 'Computing kmeans with '+str(k)+' centroids'
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters = k, verbose = False, batch_size = k * 20, compute_labels = False,reassignment_ratio = 10**-4)
    codebook.fit(D)
    cPickle.dump(codebook, open("codebook.dat", "wb"))
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    
    #Determine visual words for whole image
    init = time.time()
    print 'Computing visual words'
    visual_words = np.zeros((len(Train_descriptors), 21*k), dtype = np.float32)
#    visual_words = np.zeros((len(Train_descriptors), k), dtype = np.float32)
    #For each image, compute visual words:  we need to concatenate many histograms,
    #one for each subimage
    for i in xrange(len(Train_descriptors)):
        #Predict the words of an image
        visual_words[i, :] = spatial_pyramid.spatial_pyramid(Train_image_size[i], Train_descriptors[i], Train_keypoints[i], codebook, k)
#        words=codebook.predict(Train_descriptors[i])
#        visual_words[i,:]=np.bincount(words,minlength=k)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    
    
    # Train a linear SVM classifier
    
    stdSlr = StandardScaler().fit(visual_words) 
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
    print 'Done!'
    
    # Get all the test data and predict their labels
    visual_words_test = np.zeros((len(test_images_filenames), 21*k),dtype=np.float32)
#    visual_words_test = np.zeros((len(test_images_filenames), k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray,None)
#        words = codebook.predict(des)
#        visual_words_test[i,:] = np.bincount(words,minlength=k)
        visual_words_test[i, :] = spatial_pyramid.spatial_pyramid(np.float32(gray.shape), des, kpt, codebook, k)
    
    accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)
    
    print 'Final accuracy: ' + str(accuracy)
    
    end = time.time()
    print 'Done in ' + str(end - start) + ' secs.'
    
    ## 49.56% in 285 secs.
    
if __name__ == '__main__':
    launchsession2_pyramid()