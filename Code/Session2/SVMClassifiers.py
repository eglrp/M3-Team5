import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
import descriptors
import spatial_pyramid as spt_py
import kernel_spatial_pyr
import kernelIntersection

def trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced',probabilities=False):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight,probability = probabilities).fit(D_scaled, Train_label_per_descriptor)
    print 'Done!'

    return clf,stdSlr
 
def trainSVMKernel(visual_words,Train_label_per_descriptor,useKernelPyr,levels_pyramid,Cparam=1,probabilities=False):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    if useKernelPyr:
        kernelMatrix = kernel_spatial_pyr.spatialPyramidKernel(D_scaled,D_scaled,levels_pyramid)
    else:
        kernelMatrix = kernelIntersection.histogramIntersection(D_scaled,D_scaled)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='precomputed', C=Cparam,probability=probabilities)
    clf.fit(kernelMatrix, Train_label_per_descriptor)
    print 'Done!'
    return clf,stdSlr,D_scaled

def predict(test_images_filenames,descriptor_type,stdSlr, codebook,k, levels_pyramid, num_slots):
    #Predict test set labels with the trained classifier
    data = [codebook,k,descriptor_type, levels_pyramid]#shared data with processes
    
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    if levels_pyramid != 0:
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
    
def predictKernel(test_images_filenames,descriptor_type,clf,stdSlr,train_scaled,k,codebook,levels_pyramid,num_slots):
    #Predict test set labels with the trained classifier
    data = [train_scaled,clf,stdSlr,descriptor_type,k,codebook,levels_pyramid]#shared data with processes
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    if levels_pyramid != 0:
        predictedClasses= pool.map(getPredictionForImageKernelSpatialPyramid, test_images_filenames)
    else:
        predictedClasses= pool.map(getPredictionForImageKernel, test_images_filenames)
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
    levels_pyramid = data[3]
    
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]
    
    #Compute spatial pyramid
    visual_words = spt_py.spatial_pyramid(np.float32(gray.shape), des, coordinates_keypoints, codebook, k, levels_pyramid)
    
    return visual_words
    
def getPredictionForImageKernel(filename):
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
    test_scaled=test_scaled.reshape(1,-1)
    
    #Predict the label for each descriptor
    predictMatrix = kernelIntersection.histogramIntersection(test_scaled, train_scaled)
    SVMpredictions = computedClf.predict(predictMatrix)
    
    values, counts = np.unique(SVMpredictions, return_counts=True)
    predictedClass = values[np.argmax(counts)]
    
    return predictedClass
    
def getPredictionForImageKernelSpatialPyramid(filename):
    train_scaled=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    descriptor_type=data[3]
    k=data[4]
    codebook=data[5]
    levels_pyramid = data[6]
    
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]

    test_visual_words = spt_py.spatial_pyramid(np.float32(gray.shape), des, coordinates_keypoints, codebook, k, levels_pyramid)

    test_scaled=computedstdSlr.transform(test_visual_words)
    test_scaled=test_scaled.reshape(1,-1)
    
    #Predict the label for each descriptor
    #predictMatrix = kernelIntersection.histogramIntersection(test_scaled, train_scaled)
    predictMatrix = kernel_spatial_pyr.spatialPyramidKernel(test_scaled, train_scaled, levels_pyramid)
    SVMpredictions = computedClf.predict(predictMatrix)
    
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