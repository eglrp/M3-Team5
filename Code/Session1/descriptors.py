import cv2
import numpy as np
from abc import ABCMeta, abstractmethod

class AbstractDescriptor(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.detector=None

    @abstractmethod
    def extractKeyPointsAndDescriptors(self,grayimage):
        pass
    
    #Common to all Descriptor types
    def extractFeatures(self,train_images_filenames,train_labels,max_class_train_images):
        Train_descriptors = []
        Train_label_per_descriptor = []
        
        for i in range(len(train_images_filenames)):
            filename=train_images_filenames[i]
            if Train_label_per_descriptor.count(train_labels[i])<max_class_train_images:
                print 'Reading image '+filename
                ima=cv2.imread(filename)
                gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
                
                kpt,des=self.extractKeyPointsAndDescriptors(gray)
                
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(train_labels[i])
                print str(len(kpt))+' extracted keypoints and descriptors'
        
        # Transform everything to numpy arrays
        D=Train_descriptors[0]
        L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
        
        for i in range(1,len(Train_descriptors)):
            D=np.vstack((D,Train_descriptors[i]))
            L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))
        
        return D,L


class SIFTDescriptor(AbstractDescriptor):

    def __init__(self):
        # create the SIFT detector object
        self.detector=cv2.SIFT(nfeatures = 100)
        
    def extractKeyPointsAndDescriptors(self,grayimage):
        
        kpt,des = self.detector.detectAndCompute(grayimage,None)
        
        return kpt,des
        
class SURFDescriptor(AbstractDescriptor):
    def __init__(self):
        ## create the SURF detector object
        self.detector=cv2.SURF(hessianThreshold=100)
          
    def extractKeyPointsAndDescriptors(self,grayimage):
        kpt, des = self.detector.detectAndCompute(grayimage,None)
        
        return kpt,des
        
class ORBDescriptor(AbstractDescriptor):
    def __init__(self):
        # create the ORB detector object
        self.detector = cv2.ORB(nfeatures=100)
            
    def extractKeyPointsAndDescriptors(self,grayimage):
        kpt, des = self.detector.detectAndCompute(grayimage,None)
        
        return kpt,des