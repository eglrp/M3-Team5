import cv2
import numpy as np
import sys
from multiprocessing import Pool

#Descriptors
def getSIFTDetector():
    detector=cv2.SIFT(nfeatures=100)
    return detector
    
def getSURFDetector():
    detector=cv2.SURF(hessianThreshold=100)
    return detector

def getORBDetector():
    detector=cv2.ORB(nfeatures=100)
    return detector
    
def getKeyPointsDescriptors(detector,image):
    kpt,des=detector.detectAndCompute(image,None)
    return kpt,des

#Extract features methods
def extractFeatures(FLSubset,descriptor_type):
    data=descriptor_type
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    deslab= pool.map(getDescriptorsAndLabelsForImage, FLSubset)
    pool.terminate()

    Train_label_per_descriptor=[x[1] for x in deslab]
    Train_descriptors=[x[0] for x in deslab]
    
    # Transform everything to numpy arrays
    size_descriptors = Train_descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
        startingpoint+=len(Train_descriptors[i])
    
    return D,Train_descriptors,Train_label_per_descriptor#,L
    
def extractFeaturesOld(FLSubset,descriptor_type):
    data=descriptor_type
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    deslab= pool.map(getDescriptorsAndLabelsForImage, FLSubset)
    pool.terminate()

    Train_label_per_descriptor=[x[1] for x in deslab]
    Train_descriptors=[x[0] for x in deslab]
    
    # Transform everything to numpy arrays
    D=Train_descriptors[0]
    L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
    
    for i in range(1,len(Train_descriptors)):
        D=np.vstack((D,Train_descriptors[i]))
        L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))
    
    return D,L
    
def getDescriptorsAndLabelsForImage((filename,label)):
    print 'Reading image '+filename+' with label '+label
    descriptor_type=data
    detector=getattr(sys.modules[__name__],'get'+descriptor_type+'Detector')()
    
    ima=cv2.imread(filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)

    kpt,des=getKeyPointsDescriptors(detector,gray)

    print str(len(kpt))+' extracted keypoints and descriptors'
    return (des,label)


#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_