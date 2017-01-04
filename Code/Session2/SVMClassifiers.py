import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
import descriptors
import spatial_pyramid as spt_py

def trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced'):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight).fit(D_scaled, Train_label_per_descriptor)
    print 'Done!'

    return clf,stdSlr
    
def trainSVMOld(D,L,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced'):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(D)
    D_scaled = stdSlr.transform(D)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight).fit(D_scaled, L)
    print 'Done!'

    return clf,stdSlr

def predict(test_images_filenames,descriptor_type,stdSlr, codebook,k, Use_spatial_pyramid):
    #Predict test set labels with the trained classifier
    data = [codebook,k,descriptor_type]#shared data with processes
    
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    if Use_spatial_pyramid:
#        visual_words_test=np.zeros((len(test_images_filenames), 21*k),dtype=np.float32)
        visual_words_test = pool.map(getVisualWordsForImageSpatialPyramid, test_images_filenames)
        
        vw = np.zeros([len(visual_words_test), len(visual_words_test[0][0])], dtype = np.float32)
        for i in range(len(visual_words_test)):
            vw[i, :] = visual_words_test[0][0]
        visual_words_test = vw    
    else:
        visual_words_test=np.zeros((len(test_images_filenames),k), dtype=np.float32)
        visual_words_test= pool.map(getVisualWordsForImage, test_images_filenames)
    
    pool.terminate()

    predictedLabels = stdSlr.transform(visual_words_test)
    
    #predictions=[str(x) for x in predictedLabels]
    
    #return predictions
    return predictedLabels


def getVisualWordsForImage(filename):
    codebook=data[0]
    k=data[1]
    descriptor_type=data[2]
    
    ima=cv2.imread(filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector=getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt,des=descriptors.getKeyPointsDescriptors(detector,gray)

    #Predict the label for each descriptor
    words=codebook.predict(des)
    visual_words=np.bincount(words,minlength=k)
    
    return visual_words
    
def getVisualWordsForImageSpatialPyramid(filename):
    codebook = data[0]
    k = data[1]
    descriptor_type = data[2]
    
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray)
    coordinates_keypoints = []
    for i in range(len(kpt)):
        coordinates_keypoints.append(np.float32(kpt[i].pt))
        
    #Compute spatial pyramid    
    visual_words = spt_py.spatial_pyramid(np.float32(gray.shape), des, coordinates_keypoints, codebook, k)
    
    return visual_words    
    
def getPredictionForImageOld(filename):
    computedPca=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    descriptor_type=data[3]
    
    ima=cv2.imread(filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector=getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt,des=descriptors.getKeyPointsDescriptors(detector,gray)

    #Reduce the dimensionality of the data (PCA)
    new_des = computedPca.transform(des)
    
    #Predict the label for each descriptor
    predictions = computedClf.predict(computedstdSlr.transform(new_des))
    values, counts = np.unique(predictions, return_counts=True)
    predictedClass = values[np.argmax(counts)]
    
    return predictedClass


#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_