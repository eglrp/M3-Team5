import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
from yael import ynumpy
import descriptors
import spatial_pyramid as spt_py

def trainSVM(fisher,train_labels,Cparam=1,kernel_type='linear'):
    stdSlr = StandardScaler().fit(fisher)
    D_scaled = stdSlr.transform(fisher)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam).fit(D_scaled, train_labels)
    print 'Done!'

    return clf,stdSlr

def predict(test_images_filenames,descriptor_type,stdSlr, gmm,k, levels_pyramid,num_slots,pca):
    #Predict test set labels with the trained classifier
    data = [k,descriptor_type,gmm,levels_pyramid,pca]#shared data with processes
    
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

def getFisherForImage(filename):
    #k = data[0]
    descriptor_type = data[1]
    gmm = data[2]
    computedPca = data[4]

    kpt,des=getKptDesForImage(filename,descriptor_type)
    
    if computedPca != None:
        des#PCA
    
    fisher_test=ynumpy.fisher(gmm, des, include = ['mu','sigma'])
    
    return fisher_test

def getFisherForImageSpatialPyramid(filename):
    k = data[0]
    descriptor_type = data[1]
    gmm = data[2]
    levels_pyramid = data[3]
    computedPca = data[4]

    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    
    detector = getattr(descriptors,'get'+descriptor_type+'Detector')()
    kpt, des = descriptors.getKeyPointsDescriptors(detector,gray,descriptor_type)
    coordinates_keypoints = [kp.pt for kp in kpt]
    
    if computedPca != None:
        des#PCA

    #Compute spatial pyramid
    fisher_test = spt_py.spatial_pyramid_fisher(np.float32(gray.shape), des, coordinates_keypoints, k, gmm,levels_pyramid)

    return fisher_test

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