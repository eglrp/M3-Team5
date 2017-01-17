#!/bin/env python
import sys
sys.path.append('.')
import numpy as np
import time
import descriptors, SVMClassifiers, Evaluation, dataUtils,fisherVectors, PCA_computing

def launchsession3(num_slots,descriptor_type,randomSplits,levels_pyramid,usePCA):
    start = time.time()
    
    # Read the train and test files
    train_images_filenames,test_images_filenames,train_labels,test_labels=dataUtils.readData()
    
    #Divide training into training and validation splits
    train_percentage=0.7#70% training 30%validation
    if randomSplits:
        TrainingSplit, ValidationSplit=dataUtils.getRandomTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    else:
        TrainingSplit, ValidationSplit=dataUtils.getTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    
    #Get descriptors D
    if levels_pyramid>0:
        D, Train_descriptors, Train_label_per_descriptor, Train_keypoints, Train_image_size = descriptors.extractFeaturesPyramid(TrainingSplit,descriptor_type,num_slots)
    else:
        D, Train_descriptors, Train_label_per_descriptor = descriptors.extractFeatures(TrainingSplit, descriptor_type,num_slots)
    
    if usePCA>0:
        print 'Applying PCA'
        D, Train_descriptors, pca = PCA_computing.PCA_to_data(D, Train_descriptors, usePCA)
    else:
        pca = None
    
    #Computing gmm
    k = 64      # short codebooks (32, 64...)
    
    gmm = fisherVectors.getGMM(D,k)
    
    
    for idx,TrainDes in enumerate(Train_descriptors):    
            train_descriptor = np.float32(TrainDes)
            Train_descriptors[idx]=train_descriptor

    
    if levels_pyramid > 0:
        fisher = fisherVectors.getFisherVectorsSpatialPyramid(Train_descriptors, k, gmm, Train_image_size, Train_keypoints, levels_pyramid)
    else:
        fisher = fisherVectors.getFisherVectors(Train_descriptors,k,gmm)
        
    # Power-normalization
    #fisher=fisherVectors.powerNormalization(fisher)

    # L2 normalize
    fisher=fisherVectors.normalizeL2(fisher)
    
    # Train a linear SVM classifier
    clf, stdSlr=SVMClassifiers.trainSVM(fisher,Train_label_per_descriptor,Cparam=1,kernel_type='linear')
    
    #For test set
    # Get all the test data and predict their labels
    predictedLabels=SVMClassifiers.predict(test_images_filenames,descriptor_type,stdSlr,gmm, k, levels_pyramid,num_slots,pca)
    #Compute accuracy
    accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
    print 'Final test accuracy: ' + str(accuracy)

    #For validation set
    validation_images_filenames,validation_labels=dataUtils.unzipTupleList(ValidationSplit)
    # Get all the test data and predict their labels
    predictedLabels=SVMClassifiers.predict(validation_images_filenames,descriptor_type,stdSlr, gmm, k, levels_pyramid,num_slots,pca)
    #Compute accuracy
    validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
    print 'Final validation accuracy: ' + str(validation_accuracy)
    
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'

if __name__ == '__main__':
    num_slots=4
    levels_pyramid = 0
    randomSplits = True
    usePCA=0
    # "SIFT", "SURF", "ORB", "HARRIS", "DENSE"
    descriptor_type = "SIFT"
    print "Using %s detector, randomSplits=%s, levels_pyramid=%s, usePCA=%s" % (descriptor_type,randomSplits,levels_pyramid,usePCA)
    launchsession3(num_slots,descriptor_type,randomSplits,levels_pyramid,usePCA)