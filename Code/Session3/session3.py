#!/bin/env python
import sys
sys.path.append('.')

import time
import descriptors, SVMClassifiers, Evaluation, dataUtils,fisherVectors

def launchsession3(num_slots,descriptor_type,randomSplits,levels_pyramid,useKernelInter):
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
    
    #Computing gmm
    k = 32
    
    gmm = fisherVectors.getGMM(D,k)
    
    if levels_pyramid > 0:
        fisher = fisherVectors.getFisherVectorsSpatialPyramid(Train_descriptors, k, gmm, Train_image_size, Train_keypoints, levels_pyramid)
    else:    
        fisher = fisherVectors.getFisherVectors(Train_descriptors,k,gmm)
    
    # Train a linear SVM classifier
    if useKernelInter:
        #Kernel intersection
        clf, stdSlr,train_scaled=SVMClassifiers.trainSVMKIntersection(fisher,Train_label_per_descriptor,Cparam=1)
    else:
        clf, stdSlr=SVMClassifiers.trainSVM(fisher,Train_label_per_descriptor,Cparam=1,kernel_type='linear')
    
    #For test set
    if useKernelInter:
        predictedLabels2=SVMClassifiers.predictKernelIntersection(test_images_filenames,descriptor_type,clf,stdSlr,train_scaled,gmm,k,levels_pyramid,num_slots)
        accuracy2 = Evaluation.computeAccuracyOld(predictedLabels2,test_labels)
        print 'Final Kernel intersection test accuracy: ' + str(accuracy2)
    else:
        # Get all the test data and predict their labels
        predictedLabels=SVMClassifiers.predict(test_images_filenames,descriptor_type,stdSlr,gmm, k, levels_pyramid,num_slots)
        #Compute accuracy
        accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
        print 'Final test accuracy: ' + str(accuracy)

    #For validation set
    validation_images_filenames,validation_labels=dataUtils.unzipTupleList(ValidationSplit)
    if useKernelInter:
        #Kernel intersection
        predictedLabels2=SVMClassifiers.predictKernelIntersection(validation_images_filenames,descriptor_type,clf,stdSlr,train_scaled,gmm,k,levels_pyramid,num_slots)
        accuracy2 = Evaluation.computeAccuracyOld(predictedLabels2,validation_labels)
        print 'Final Kernel intersection validation accuracy: ' + str(accuracy2)
    else:
        # Get all the test data and predict their labels
        predictedLabels=SVMClassifiers.predict(validation_images_filenames,descriptor_type,stdSlr, gmm, k, levels_pyramid,num_slots)
        #Compute accuracy
        validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)
    
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    
    ## 61.71% in 251 secs.
    
if __name__ == '__main__':
    num_slots=4
    levels_pyramid = 0
    useKernelInter = False
    randomSplits = True
    # "SIFT", "SURF", "ORB", "HARRIS", "DENSE"
    descriptor_type = "SIFT"
    print "Using %s detector, randomSplits=%s, levels_pyramid=%s, useKernelInter=%s" % (descriptor_type,randomSplits,levels_pyramid,useKernelInter)
    launchsession3(num_slots,descriptor_type,randomSplits,levels_pyramid,useKernelInter)