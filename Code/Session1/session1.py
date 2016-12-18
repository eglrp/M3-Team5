#!/bin/env python
import sys
sys.path.append('.')
import time
import inputOutputUtils
import Evaluation
import PCA_computing
from descriptors import SIFTDescriptor,SURFDescriptor,ORBDescriptor
from SVMClassifiers import SVMClassifier
def launchsession1():
    start = time.time()
    
    # Read the train and test files
    train_images_filenames,test_images_filenames,train_labels,test_labels=inputOutputUtils.readData()
    
    # Create Descriptors (SIFT, SURF, etc)
    mySIFTDescriptor=SIFTDescriptor();#SIFT Descriptor
    #mySURFDescriptor = SURFDescriptor(); # SURF Descriptor
    #myORBDescriptor = ORBDescriptor(); # ORB Descriptor

    # Obtain descriptors and labels for the training set
    max_class_train_images=30
    D,L=mySIFTDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)
    #D,L=mySURFDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)
    #D,L=myORBDescriptor.extractFeatures(train_images_filenames,train_labels,max_class_train_images)
    
    #Apply PCA to descriptors
    print 'Applying PCA'
    number_components = 90
    D, pca = PCA_computing.PCA_to_data(D, number_components)
    
    # Train a SVM classifier
    mySVMClassifier=SVMClassifier(C=1,kernel_type='linear')#SVMClassifier with C=1 and linear Kernel
    #mySVMClassifier=SVMClassifier(C=1,kernel_type='rbf')
    #mySVMClassifier=SVMClassifier(C=1,kernel_type='poly',degree_value = 1, gamma_value = 0.01,weight = 'balanced')

    clf, stdSlr=mySVMClassifier.train(D,L)

    
    # Get all the test data and predict their labels
    predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,mySIFTDescriptor,clf,stdSlr, pca)
    #predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,mySURFDescriptor,clf,stdSlr, pca)
    #predictedClasses=mySVMClassifier.predict(test_images_filenames,test_labels,myORBDescriptor,clf,stdSlr, pca)
    
    
    # Performance evaluation
    accuracy=Evaluation.computeAccuracy(predictedClasses,test_labels)

    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    
    f = open("output.txt", 'w')
    f.write(str(accuracy))
    f.close()
    
    ## 38.78% in 797 secs.
    
    
if __name__ == '__main__':
    launchsession1()
