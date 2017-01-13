import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
import descriptors
import spatial_pyramid as spt_py
import kernelIntersection

def trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced',probabilities=False):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight,probability = probabilities).fit(D_scaled, Train_label_per_descriptor)
    print 'Done!'

    return clf,stdSlr
 
def trainSVMKIntersection(visual_words,Train_label_per_descriptor,Cparam=1,probabilities=False):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    kernelMatrix =kernelIntersection.histogramIntersection(D_scaled,D_scaled)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='precomputed', C=Cparam,probability=probabilities)
    #temp = kernelMatrix.reshape(1,-1)
    temp=np.tile(kernelMatrix, (len(kernelMatrix), 1))
    #clf.fit(kernelMatrix, Train_label_per_descriptor)
    clf.fit(temp, Train_label_per_descriptor)
    print 'Done!'
    return clf,stdSlr,D_scaled

def predict(test_images_filenames,descriptor_type,stdSlr, codebook,k, Use_spatial_pyramid, num_slots):
    #Predict test set labels with the trained classifier
    data = [codebook,k,descriptor_type, Use_spatial_pyramid]#shared data with processes
    
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    if Use_spatial_pyramid != 0:
#        visual_words_test=np.zeros((len(test_images_filenames), 21*k),dtype=np.float32)
        visual_words_test = pool.map(getVisualWordsForImageSpatialPyramid, test_images_filenames)
        
        vw = np.zeros([len(visual_words_test), len(visual_words_test[0][0])], dtype = np.float32)
        for i in range(len(visual_words_test)):
            vw[i, :] = visual_words_test[i][0]
        visual_words_test = vw    
    else:
        visual_words_test=np.zeros((len(test_images_filenames),k), dtype=np.float32)
        visual_words_test= pool.map(getVisualWordsForImage, test_images_filenames)
    
    pool.terminate()

    predictedLabels = stdSlr.transform(visual_words_test)
    
    #predictions=[str(x) for x in predictedLabels]
    
    #return predictions
    return predictedLabels
    
def predictKernelIntersection(test_images_filenames,descriptor_type,clf,stdSlr,train_scaled,k,codebook,Use_spatial_pyramid,num_slots):
    #Predict test set labels with the trained classifier
    data = [train_scaled,clf,stdSlr,descriptor_type,k,codebook]#shared data with processes
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    if Use_spatial_pyramid:
        predictedClasses= pool.map(getPredictionForImageKIntersectionSpatialPyramid, test_images_filenames)
    else:
        predictedClasses= pool.map(getPredictionForImageKIntersection, test_images_filenames)
    
    pool.terminate()
    
    predictions=[str(x) for x in predictedClasses]
    
    return predictions

def getVisualWordsForImage(filename):
    codebook=data[0]
    k=data[1]
    descriptor_type=data[2]
    
    kpt,des=getKptDesForImage(filename,descriptor_type)

    #Predict the label for each descriptor
    words=codebook.predict(des)
    visual_words=np.bincount(words,minlength=k)
    
    return visual_words
    
def getVisualWordsForImageSpatialPyramid(filename):
    codebook = data[0]
    k = data[1]
    descriptor_type = data[2]
    Use_spatial_pyramid = data[3]
    
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]
    
    #Compute spatial pyramid
    visual_words = spt_py.spatial_pyramid(np.float32(gray.shape), des, coordinates_keypoints, codebook, k, Use_spatial_pyramid)
    
    return visual_words
    
def getPredictionForImageKIntersection(filename):
    train_scaled=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    descriptor_type=data[3]
    k=data[4]
    codebook=data[5]
    
    kpt,des=getKptDesForImage(filename,descriptor_type)
    
    words=codebook.predict(des)
    test_visual_words=np.bincount(words,minlength=k)
    
    test_scaled=computedstdSlr.transform(test_visual_words)
    tem=test_scaled.reshape(1,-1)
    temp=np.tile(tem, (len(train_scaled), 1))
    
    
    #Predict the label for each descriptor
    #predictMatrix = kernelIntersection.histogramIntersection(test_scaled, train_scaled)
    predictMatrix = kernelIntersection.histogramIntersection(temp, train_scaled)

    
    temp=np.tile(predictMatrix, (len(train_scaled), 1))
    
    SVMpredictions = computedClf.predict(temp)
    
    values, counts = np.unique(SVMpredictions, return_counts=True)
    predictedClass = values[np.argmax(counts)]
    
    return predictedClass
    
def getPredictionForImageKIntersectionSpatialPyramid(filename):
    train_scaled=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    descriptor_type=data[3]
    k=data[4]
    codebook=data[5]
    
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]

    test_visual_words = spt_py.spatial_pyramid(np.float32(gray.shape), des, coordinates_keypoints, codebook, k)
    
    test_scaled=computedstdSlr.transform(test_visual_words)
    tem=test_scaled.reshape(1,-1)
    temp=np.tile(tem, (len(train_scaled), 1))
    
    
    #Predict the label for each descriptor
    predictMatrix = kernelIntersection.histogramIntersection(temp, train_scaled)

    
    temp=np.tile(predictMatrix, (len(train_scaled), 1))
    
    SVMpredictions = computedClf.predict(temp)
    
    values, counts = np.unique(SVMpredictions, return_counts=True)
    predictedClass = values[np.argmax(counts)]
    
    return predictedClass

def getKptDesForImage(filename,descriptor_type):
    ima=cv2.imread(filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector=getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt,des=descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    
    return kpt,des
    
#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_