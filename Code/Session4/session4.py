# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:11:25 2017

@author: onofre
"""

#!/bin/env python
import sys
sys.path.append('.')

import time
import descriptors, SVMClassifiers, Evaluation, dataUtils,BoW,graphs
import matplotlib.pyplot as plt
def launchsession4(num_slots, descriptor_type, randomSplits,levels_pyramid, useKernelInter, useKernelPyr, rocCurveCM):
    start = time.time()
    
    # Read the train and test files
    train_images_filenames, test_images_filenames, train_labels, test_labels = dataUtils.readData()
    
    #Divide training into training and validation splits
    train_percentage = 0.7#70% training 30%validation
    if randomSplits:
        TrainingSplit, ValidationSplit=dataUtils.getRandomTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    else:
        TrainingSplit, ValidationSplit=dataUtils.getTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    
    #Obtain information from VGG ConvNet
    if levels_pyramid != 0:
        D, Train_descriptors, Train_label_per_descriptor, Train_keypoints, Train_image_size = descriptors.extractFeaturesPyramid(TrainingSplit,descriptor_type,num_slots)
    else:
        D, Train_descriptors, Train_label_per_descriptor = descriptors.extractFeatures(TrainingSplit, descriptor_type,num_slots)
    
    #Computing bag of words using k-means and save codebook
    k = 512
    codebook=BoW.computeCodebook(D,k)

    #Determine visual words
    if levels_pyramid != 0:
        visual_words = BoW.getVisualWordsSpatialPyramid(codebook, k, Train_descriptors, Train_image_size, Train_keypoints, levels_pyramid)
    else:    
        visual_words = BoW.getVisualWords(codebook, k, Train_descriptors)
    
    # Train a linear SVM classifier
    if useKernelInter|useKernelPyr:
        #Kernel intersection
        clf, stdSlr,train_scaled=SVMClassifiers.trainSVMKernel(visual_words,Train_label_per_descriptor,useKernelPyr,levels_pyramid,Cparam=1,probabilities=rocCurveCM)
    else:
        clf, stdSlr=SVMClassifiers.trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear',probabilities=rocCurveCM)


    #For test set
    if useKernelInter|useKernelPyr:
        predictedLabels2 = SVMClassifiers.predictKernel(test_images_filenames,descriptor_type,clf,stdSlr,train_scaled,k,codebook,levels_pyramid,num_slots)
        accuracy2 = Evaluation.computeAccuracyOld(predictedLabels2,test_labels)
        print 'Final Kernel intersection test accuracy: ' + str(accuracy2)
    else:
        # Get all the test data and predict their labels
        predictedLabels=SVMClassifiers.predict(test_images_filenames,descriptor_type,stdSlr, codebook, k, levels_pyramid,num_slots)
        #Compute accuracy
        accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
        print 'Final test accuracy: ' + str(accuracy)

    #For validation set
    validation_images_filenames, validation_labels = dataUtils.unzipTupleList(ValidationSplit)
    if useKernelInter|useKernelPyr:
        predictedLabels2=SVMClassifiers.predictKernel(validation_images_filenames,descriptor_type,clf,stdSlr,train_scaled,k,codebook,levels_pyramid,num_slots)
        accuracy2 = Evaluation.computeAccuracyOld(predictedLabels2,validation_labels)
        print 'Final Kernel intersection validation accuracy: ' + str(accuracy2)
    else:
        # Get all the test data and predict their labels
        predictedLabels=SVMClassifiers.predict(validation_images_filenames,descriptor_type,stdSlr, codebook, k, levels_pyramid,num_slots)
        #Compute accuracy
        validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)
    #Roc curve and Confusion Matrix
    if rocCurveCM:
        graphs.rcurve(predictedLabels,validation_labels,clf)
        graphs.plot_confusion_matrix(clf,validation_labels,stdSlr.transform(predictedLabels),normalize=False,title='Confusion matrix',cmap=plt.cm.Blues)
    
    end = time.time()
    print 'Done in '+str(end-start)+' secs.'


if __name__ == '__main__':
    num_slots = 4
    useKernelInter = False
    randomSplits = True
    levels_pyramid = 0
    useKernelPyr = False
    rocCurveCM = False

    # "SIFT", "SURF", "ORB", "HARRIS", "DENSE"
    layer_taken = "SIFT"
    print "Taking layer %s , randomSplits=%s, levels_pyramid=%s, useKernelInter=%s, useKernelPyr=%s" % (layer_taken, randomSplits, levels_pyramid, useKernelInter, useKernelPyr)
    launchsession4(num_slots, layer_taken, randomSplits, levels_pyramid, useKernelInter, useKernelPyr, rocCurveCM)