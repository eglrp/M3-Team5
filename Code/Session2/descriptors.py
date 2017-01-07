import cv2
import numpy as np
import sys
from multiprocessing import Pool

#Detectors
def getSIFTDetector():
    detector=cv2.SIFT(nfeatures=100)
    return detector
    
def getSURFDetector():
    detector=cv2.SURF(hessianThreshold=400)
    return detector

def getORBDetector():
    detector=cv2.ORB(nfeatures=100)
    return detector

def getHARRISDetector():
    detector = cv2.FeatureDetector_create("HARRIS")
    return detector
     
#def getMSERDetector():
#    #detector = cv2.MSER()
#    detector=cv2.FeatureDetector_create("MSER")
#    return detector

#def getSTARDetector():
#    #detector = cv2.StarDetector()
#    detector = cv2.StarFeatureDetector()
#    #detector = cv2.FeatureDetector_create("STAR")
#    return detector

#def getFASTDetector():
#    detector = cv2.FastFeatureDetector(nonmaxSuppression=0)
#    return detector    

def getDENSEDetector():
#    detector=cv2.DenseFeatureDetector(initFeatureScale=30, featureScaleLevels=1,
#                          featureScaleMul=1,
#                          initXyStep=10, initImgBound=10,
#                          varyXyStepWithScale=0,
#                          varyImgBoundWithScale=0)
    detector = cv2.FeatureDetector_create("DENSE")
    detector.setInt("initXyStep",10)
    detector.setInt("initFeatureScale",30)
    detector.setInt("initImgBound",10)

#    detector = cv2.FeatureDetector_create("DENSE")
#    detector.setInt("initXyStep",10)
#    detector.setInt("initFeatureScale",30)
#    detector.setInt("initImgBound",10)
    detector = None
    
    return detector
    

    
def getKeyPointsDescriptors(detector,image,descriptor_type):
    #Descriptors
    if descriptor_type == 'SIFT' :
        kpt,des=detector.detectAndCompute(image,None)
    elif descriptor_type=='DENSE':
        
        sift=getSIFTDetector()
        kp1 = np.zeros((image.shape[0]*image.shape[1])/100)
        k=0
        for x in xrange(int(0,image.shape[0],10)):
            for y in xrange(int(0,image.shape[1],10)):
                kp1[k] = cv2.KeyPoint([x,y,np.random.randint(10, 30)])   
                k=k+1
                print(k)
        kpt,des=sift.compute(image,kp1)
       
    else:
        sift=getSIFTDetector()
        kp1=detector.detect(image,None)
        kpt,des=sift.compute(image,kp1)
        if descriptor_type == 'SURF':
            kpt=kpt[0:100]
            des=des[0:100,:]
    return kpt,des

#Extract features methods
def extractFeatures(FLSubset, descriptor_type):
    data=descriptor_type
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    deslab = pool.map(getDescriptorsAndLabelsForImage, FLSubset)
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
    
def extractFeaturesPyramid(FLSubset, descriptor_type):
    data=descriptor_type
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    deslab = pool.map(getDescriptorsAndLabelsForImage, FLSubset)
    pool.terminate()

    Train_label_per_descriptor=[x[1] for x in deslab]
    Train_descriptors=[x[0] for x in deslab]
    Train_keypoints = [x[2] for x in deslab]
    Train_image_size= [x[3] for x in deslab]
    
    size_descriptors = Train_descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
        startingpoint+=len(Train_descriptors[i])
    
    return D, Train_descriptors, Train_label_per_descriptor, Train_keypoints, Train_image_size
    
def getDescriptorsAndLabelsForImage((filename,label)):
    print 'Reading image '+filename+' with label '+label
    descriptor_type=data
    detector=getattr(sys.modules[__name__],'get'+descriptor_type+'Detector')()
    
    ima=cv2.imread(filename)
    gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)

    kpt, des=getKeyPointsDescriptors(detector,gray,descriptor_type)
    
    coordinates_keypoints = [kp.pt for kp in kpt]
    print str(len(kpt))+' extracted keypoints and descriptors'
#    return (des, label)
    return (des, label, coordinates_keypoints, np.float32(gray.shape))

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_