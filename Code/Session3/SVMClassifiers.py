import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
from yael import ynumpy
import descriptors
import spatial_pyramid as spt_py
import kernelIntersection

def trainSVM(fisher,train_labels,Cparam=1,kernel_type='linear'):
    stdSlr = StandardScaler().fit(fisher)

    D_scaled = stdSlr.transform(fisher)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam).fit(D_scaled, train_labels)
    print 'Done!'

    return clf,stdSlr

def trainSVMKIntersection(fisher,train_labels,Cparam=1):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(fisher)
    D_scaled = stdSlr.transform(fisher)
    kernelMatrix =kernelIntersection.histogramIntersection(D_scaled,D_scaled)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='precomputed', C=Cparam)
    clf.fit(kernelMatrix, train_labels)
    print 'Done!'
    return clf,stdSlr,D_scaled

def predict(test_images_filenames,descriptor_type,stdSlr, gmm,k, levels_pyramid,num_slots):
    #Predict test set labels with the trained classifier
    data = [k,descriptor_type,gmm,levels_pyramid]#shared data with processes
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    if levels_pyramid > 0:
        fisher_test = pool.map(getFisherForImageSpatialPyramid, test_images_filenames)
        fv = np.zeros([len(fisher_test), len(fisher_test[0][0])], dtype = np.float32)
        for i in range(len(fisher_test)):
            fv[i, :] = fisher_test[i][0]
        fisher_test = fv
        
    else:
        fisher_test= pool.map(getFisherForImage, test_images_filenames)
    pool.terminate()

    predictedLabels = stdSlr.transform(fisher_test)
    return predictedLabels
    
def predictKernelIntersection(test_images_filenames,descriptor_type,clf,stdSlr,train_scaled,gmm,k,levels_pyramid,num_slots):
    #Predict test set labels with the trained classifier
    data = [train_scaled,clf,stdSlr,descriptor_type,k,gmm,levels_pyramid]#shared data with processes
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    if levels_pyramid>0:
        predictedClasses= pool.map(getPredictionForImageKIntersectionSpatialPyramid, test_images_filenames)
    else:
        predictedClasses= pool.map(getPredictionForImageKIntersection, test_images_filenames)
    pool.terminate()
    
    predictions=[str(x) for x in predictedClasses]
    
    return predictions

def getFisherForImage(filename):
    #k = data[0]
    descriptor_type = data[1]
    gmm = data[2]

    kpt,des=getKptDesForImage(filename,descriptor_type)
    
    fisher_test=ynumpy.fisher(gmm, des, include = ['mu','sigma'])
    
    return fisher_test
    
def getFisherForImageSpatialPyramid(filename):
    k = data[0]
    descriptor_type = data[1]
    gmm = data[2]
    levels_pyramid = data[3]

    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]
    
    #Compute spatial pyramid
    fisher_test = spt_py.spatial_pyramid_fisher(np.float32(gray.shape), des, coordinates_keypoints, k, gmm,levels_pyramid)
                                                
    return fisher_test
    
def getPredictionForImageKIntersection(filename):
    train_scaled=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    descriptor_type=data[3]
    k=data[4]
    gmm=data[5]

    kpt,des=getKptDesForImage(filename,descriptor_type)
    fisher_test=ynumpy.fisher(gmm, des, include = ['mu','sigma'])
    test_scaled=computedstdSlr.transform(fisher_test)
    test_scaled=test_scaled.reshape(1,-1)
    
    predictMatrix = kernelIntersection.histogramIntersection(test_scaled, train_scaled)
    SVMpredictions = computedClf.predict(predictMatrix)
    
    values, counts = np.unique(SVMpredictions, return_counts=True)
    predictedClass = values[np.argmax(counts)]
    
    return predictedClass
    
def getPredictionForImageKIntersectionSpatialPyramid(filename):
    train_scaled=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    descriptor_type=data[3]
    k=data[4]
    gmm=data[5]
    levels_pyramid=data[6]
    
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]

    fisher_test = spt_py.spatial_pyramid_fisher(np.float32(gray.shape), des, coordinates_keypoints, k, gmm, levels_pyramid)
    
    
    test_scaled=computedstdSlr.transform(fisher_test)
    test_scaled=test_scaled.reshape(1,-1)
    
    predictMatrix = kernelIntersection.histogramIntersection(test_scaled, train_scaled)
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