#!/bin/env python
import sys
sys.path.append('.')

import time
import SVMClassifiers, Evaluation, dataUtils,BoW, descriptors, PCA_computing

<<<<<<< HEAD
def launchsession4(num_slots, layer_taken, randomSplits, k, useServer,usePCA):
=======
def launchsession4(num_slots, layer_taken, randomSplits, k, useServer, method_used):
>>>>>>> 1fbc031b361b5ab9ac9d88672d1b27a00c612381
    start = time.time()
    
    # Read the train and test files
    if useServer:
        train_images_filenames, test_images_filenames, train_labels, test_labels = dataUtils.readServerData()
    else:
        train_images_filenames, test_images_filenames, train_labels, test_labels = dataUtils.readData()
    
    #For testing with smaller database
#    train_images_filenames=train_images_filenames[:300]
#    test_images_filenames=test_images_filenames[:300]
#    train_labels=train_labels[:300]
#    test_labels=test_labels[:300]
    
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
    
    if usePCA>0:
        print 'Applying PCA'
        D, Train_descriptors, pca = PCA_computing.PCA_to_data(D, Train_descriptors, usePCA)
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
<<<<<<< HEAD
        predictedLabels=SVMClassifiers.predictBoVW(TestSplit, layer_taken, stdSlr, codebook, k, CNN_base_model, num_slots,pca)
=======
        predictedLabels=SVMClassifiers.predictBoVW(TestSplit, layer_taken, stdSlr, codebook, k, CNN_base_model, num_slots, method_used)
>>>>>>> 1fbc031b361b5ab9ac9d88672d1b27a00c612381
        accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,test_labels)
        print 'Final test accuracy: ' + str(accuracy)

    #For validation set
<<<<<<< HEAD
    print 'Predicting labels for validation set'
=======
    validation_images_filenames, validation_labels = dataUtils.unzipTupleList(ValidationSplit)
>>>>>>> afa2cde972edae148ceaf3c903da3529f67c6cef
    if layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten':
        #Not using BoVW
        predictedLabels=SVMClassifiers.predict(ValidationSplit, layer_taken, stdSlr, clf, CNN_base_model, num_slots)
        validation_accuracy = Evaluation.computeAccuracyOld(predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)
    else:
        #BoVW
<<<<<<< HEAD
        validation_images_filenames, validation_labels = dataUtils.unzipTupleList(ValidationSplit)
<<<<<<< HEAD
        predictedLabels = SVMClassifiers.predictBoVW(ValidationSplit, layer_taken, stdSlr, codebook, k,CNN_base_model, num_slots,pca)
=======
        predictedLabels = SVMClassifiers.predictBoVW(ValidationSplit, layer_taken, stdSlr, codebook, k,CNN_base_model, num_slots, method_used)
=======
        predictedLabels = SVMClassifiers.predictBoVW(ValidationSplit, layer_taken, stdSlr, codebook, k,CNN_base_model, num_slots)
>>>>>>> afa2cde972edae148ceaf3c903da3529f67c6cef
>>>>>>> 1fbc031b361b5ab9ac9d88672d1b27a00c612381
        validation_accuracy = Evaluation.getMeanAccuracy(clf,predictedLabels,validation_labels)
        print 'Final validation accuracy: ' + str(validation_accuracy)

    end = time.time()
    print 'Done in '+str(end-start)+' secs.'


if __name__ == '__main__':
    num_slots = 2
    randomSplits = False
    useServer=False
    method_used = {'method_to_reduce_dim': 'PCA', 'Value_PCA':90}
    method_used = {'clear_zero_features': True}
    method_used = {'method_to_reduce_dim': 'average', 'Remaining_features':100}
    layer_taken = "block5_pool"# Layer
<<<<<<< HEAD
    k = 512 #Centroids for BoVW codebook
    usePCA=0
    
    print "Taking layer %s , randomSplits = %s, k-means centroids: %s" % (layer_taken, randomSplits, k)
    launchsession4(num_slots, layer_taken, randomSplits, k, useServer,usePCA)
=======
    k = 128 #Centroids for BoVW codebook

    print "Taking layer %s , randomSplits = %s, k-means centroids: %s" % (layer_taken, randomSplits, k)
    launchsession4(num_slots, layer_taken, randomSplits, k, useServer, method_used)
>>>>>>> 1fbc031b361b5ab9ac9d88672d1b27a00c612381
