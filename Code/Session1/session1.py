#!/bin/env python
import sys
sys.path.append('.')
import time
import inputOutputUtils
import Evaluation
import PCA_computing
import descriptors, SVMClassifiers

def launchsession1():
    start = time.time()
    
    # Read the train and test files
    train_images_filenames,test_images_filenames,train_labels,test_labels=inputOutputUtils.readData()
    
    #Descriptor: SIFT, ORB, SURF
    descriptor_type='SIFT'
    
    #subset of certain number of images per class
    max_class_train_images=30
    FLSubset=inputOutputUtils.getFilenamesLabelsSubset(train_images_filenames,train_labels,max_class_train_images)
    
    # Obtain descriptors and labels for the training set
    D,L=descriptors.extractFeatures(FLSubset,descriptor_type)#it can be ORB or SURF
    
    #Apply PCA to descriptors
    print 'Applying PCA'
    number_components = 90
    D, pca = PCA_computing.PCA_to_data(D, number_components)
    
    # Train a SVM classifier
    clf, stdSlr=SVMClassifiers.trainSVM(D,L,Cparam=1,kernel_type='linear')
    #Other options:
    #(D,L,Cparam=1,kernel_type='rbf')
    #(D,L,Cparam=1,kernel_type='poly',degree_value = 1, gamma_value = 0.01,weight = 'balanced')

    
    # Get all the test data and predict their labels
    predictedClasses=SVMClassifiers.predict(test_images_filenames,descriptor_type,clf,stdSlr, pca)

    # Performance evaluation
    accuracy=Evaluation.computeAccuracy(predictedClasses,test_labels)
    
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    
    f = open("output.txt", 'w')
    f.write(str(accuracy))
    f.close()

if __name__ == '__main__':
    launchsession1()
