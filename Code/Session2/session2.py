#!/bin/env python
import sys
sys.path.append('.')

import time
import descriptors, SVMClassifiers, Evaluation, dataUtils,BoW

def launchsession2():
    start = time.time()
    Use_spatial_pyramid = False
    useKernelInter = False
    
    # Read the train and test files
    train_images_filenames,test_images_filenames,train_labels,test_labels=dataUtils.readData()
    
    #Divide training into training and validation splits
    train_percentage=0.7#70% training 30%validation
    TrainingSplit, ValidationSplit=dataUtils.getRandomTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    
    #Get descriptors D
    if Use_spatial_pyramid:
        D, Train_descriptors, Train_label_per_descriptor, Train_keypoints, Train_image_size = descriptors.extractFeaturesPyramid(TrainingSplit,'SIFT')
    else:
        D, Train_descriptors, Train_label_per_descriptor = descriptors.extractFeatures(TrainingSplit, 'SIFT')
    
    #Computing bag of words using k-means and save codebook
    k = 512
    codebook=BoW.computeCodebook(D,k)

    #Determine visual words
    if Use_spatial_pyramid:
        visual_words = BoW.getVisualWordsSpatialPyramid(codebook, k, Train_descriptors, Train_image_size, Train_keypoints)
    else:    
        visual_words = BoW.getVisualWords(codebook, k, Train_descriptors)
    
    # Train a linear SVM classifier
    if useKernelInter:
        #Kernel intersection
        clf, stdSlr,train_scaled=SVMClassifiers.trainSVMKIntersection(visual_words,Train_label_per_descriptor,Cparam=1)
    else:
        clf, stdSlr=SVMClassifiers.trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear')


    #For test set
    if useKernelInter:
        predictedLabels2=SVMClassifiers.predictKernelIntersection(test_images_filenames,'SIFT',clf,stdSlr,train_scaled,k,codebook,Use_spatial_pyramid)
        accuracy2 = Evaluation.computeAccuracyOld(predictedLabels2,test_labels)
        print 'Final Kernel intersection test accuracy: ' + str(accuracy2)
    else:
        # Get all the test data and predict their labels
        predictedLabels=SVMClassifiers.predict(test_images_filenames,'SIFT',stdSlr, codebook, k, Use_spatial_pyramid)
        #Compute accuracy
        accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
        print 'Final test accuracy: ' + str(accuracy)

    #For validation set
    validation_images_filenames,validation_labels=dataUtils.unzipTupleList(ValidationSplit)
    if useKernelInter:
        #Kernel intersection
        predictedLabels2=SVMClassifiers.predictKernelIntersection(validation_images_filenames,'SIFT',clf,stdSlr,train_scaled,k,codebook,Use_spatial_pyramid)
        accuracy2 = Evaluation.computeAccuracyOld(predictedLabels2,validation_labels)
        print 'Final Kernel intersection validation accuracy: ' + str(accuracy2)
    else:
        # Get all the test data and predict their labels
        predictedLabels=SVMClassifiers.predict(validation_images_filenames,'SIFT',stdSlr, codebook, k, Use_spatial_pyramid)
        #Compute accuracy
        validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)
    
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    
    ## 49.56% in 285 secs.
    
if __name__ == '__main__':
    launchsession2()