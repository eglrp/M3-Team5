#!/bin/env python
import sys
sys.path.append('.')

import time
import SVMClassifiers, Evaluation, dataUtils,BoW, descriptors, PCA_computing


def launchsession4(num_slots, layer_taken, randomSplits, k, useServer, method_used):

    start = time.time()
    
    # Read the train and test files
    if useServer:
        train_images_filenames, test_images_filenames, train_labels, test_labels = dataUtils.readServerData()
    else:
        train_images_filenames, test_images_filenames, train_labels, test_labels = dataUtils.readData()
    
    #For testing with smaller database
    rr = 4
    train_images_filenames=train_images_filenames[116: 116 + rr]
    test_images_filenames=test_images_filenames[116: 116 + rr]
    train_labels=train_labels[116: 116 + rr]
    test_labels=test_labels[116: 116 + rr]
    
    #Divide training into training and validation splits
    train_percentage = 0.7 #70% training 30%validation
    if randomSplits:
        TrainingSplit, ValidationSplit=dataUtils.getRandomTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    else:
        TrainingSplit, ValidationSplit=dataUtils.getTrainingValidationSplit(train_images_filenames,train_labels,train_percentage)
    
    #Obtain information from VGG ConvNet
    CNN_base_model = descriptors.getBaseModel()#Base model
    
                                             
    #Compute features
    print 'Extracting features'
    D, Train_descriptors, Train_label_per_descriptor = descriptors.extractFeaturesMaps(TrainingSplit, layer_taken, CNN_base_model, num_slots, method_used)
    
    if  method_used['usePCA'] > 0:
        print 'Applying PCA'
        D, Train_descriptors, pca = PCA_computing.PCA_to_data(D, Train_descriptors, method_used['usePCA'])
    else:
        pca = None

    if layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten':
        visual_words = D
        codebook = None
    else:
        #Computing bag of words using k-means and save codebook when necessary
        codebook = BoW.computeCodebook(D, k)
        #Determine visual words
        visual_words = BoW.getVisualWords(codebook, k, Train_descriptors)
    
    # Train a linear SVM classifier
    clf, stdSlr = SVMClassifiers.trainSVM(visual_words, Train_label_per_descriptor,Cparam=1,kernel_type='linear')

    
    #For test set
    print 'Predicting labels for test set'
    TestSplit=zip(test_images_filenames,test_labels)
    if layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten':
        ##Not using BoVW
        predictedLabels=SVMClassifiers.predict(TestSplit, layer_taken, stdSlr, clf, CNN_base_model, num_slots)
        accuracy = Evaluation.computeAccuracyOld(predictedLabels,test_labels)
        print 'Final test accuracy: ' + str(accuracy)
    else:
        #BoVW
        predictedLabels=SVMClassifiers.predictBoVW(TestSplit, layer_taken, stdSlr, codebook, k, CNN_base_model, num_slots, pca , method_used)
        accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
        print 'Final test accuracy: ' + str(accuracy)

    
    #For validation set
    print 'Predicting labels for validation set'
    validation_images_filenames, validation_labels = dataUtils.unzipTupleList(ValidationSplit)
    if layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten':
        #Not using BoVW
        predictedLabels=SVMClassifiers.predict(ValidationSplit, layer_taken, stdSlr, clf, CNN_base_model, num_slots)
        validation_accuracy = Evaluation.computeAccuracyOld(predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)
    else:
        #BoVW
        validation_images_filenames, validation_labels = dataUtils.unzipTupleList(ValidationSplit)
        predictedLabels = SVMClassifiers.predictBoVW(ValidationSplit, layer_taken, stdSlr, codebook, k,CNN_base_model, num_slots, pca , method_used)




        predictedLabels = SVMClassifiers.predictBoVW(ValidationSplit, layer_taken, stdSlr, codebook, k,CNN_base_model, num_slots)

        validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)

    end = time.time()
    print 'Done in '+str(end-start)+' secs.'


if __name__ == '__main__':
    num_slots = 2
    randomSplits = False
    useServer=False

    method_used = {'method_to_reduce_dim': 'Nothing', 'Remaining_features': 100, 'clear_zero_features': True, 'usePCA': 30}
    layer_taken = "block5_pool"# Layer

    k = 128 #Centroids for BoVW codebook

    print "Taking layer %s , randomSplits = %s, k-means centroids: %s" % (layer_taken, randomSplits, k)
    launchsession4(num_slots, layer_taken, randomSplits, k, useServer, method_used)

