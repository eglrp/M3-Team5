#!/bin/env python
import sys
sys.path.append('.')

import time
import SVMClassifiers, Evaluation, dataUtils,BoW


def launchsession4(num_slots, layer_taken, randomSplits):
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
    
    
    
    
    
    #Computing bag of words using k-means and save codebook
    k = 512
    codebook=BoW.computeCodebook(D,k)

    #Determine visual words
    visual_words = BoW.getVisualWords(codebook, k, Train_descriptors)
    
    # Train a linear SVM classifier
    clf, stdSlr=SVMClassifiers.trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear')


    #For test set
    # Get all the test data and predict their labels
    predictedLabels=SVMClassifiers.predict(test_images_filenames,stdSlr, codebook, k,num_slots)
    #Compute accuracy
    accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
    print 'Final test accuracy: ' + str(accuracy)

    #For validation set
    validation_images_filenames, validation_labels = dataUtils.unzipTupleList(ValidationSplit)
    # Get all the test data and predict their labels
    predictedLabels=SVMClassifiers.predict(validation_images_filenames,stdSlr, codebook, k,num_slots)
    #Compute accuracy
    validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
    print 'Final validation accuracy: ' + str(validation_accuracy)
    
    end = time.time()
    print 'Done in '+str(end-start)+' secs.'


if __name__ == '__main__':
    num_slots = 4
    randomSplits = False
    # Layer
    layer_taken = "fc1"
    
    print "Taking layer %s , randomSplits=%s" % (layer_taken, randomSplits)
    launchsession4(num_slots, layer_taken, randomSplits)