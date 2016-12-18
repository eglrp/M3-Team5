import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

class SVMClassifier(object):
    #Parameters of the SVM
    def __init__(self,C,kernel_type, degree_value,gamma_value,weight):
        self.Cparam=C
        self.kernel=kernel_type
        self.degree=degree_value
        self.gamma=gamma_value
        self.weight=weight
        
    # Train a linear SVM classifier
    def train(self,D,L):
        stdSlr = StandardScaler().fit(D)
        D_scaled = stdSlr.transform(D)
        print 'Training the SVM classifier...'
        clf = svm.SVC(kernel=self.kernel, C=self.Cparam, degree=self.degree, gamma = self.gamma,class_weight = self.weight).fit(D_scaled, L)
        print 'Done!'
        
        return clf,stdSlr
        
    #Predict test set labels with the trained classifier
    def predict(self,test_images_filenames,test_labels,Descriptor,clf,stdSlr, pca):
        predictedClasses=np.empty(len(test_images_filenames),dtype='|S15')
        for i in range(len(test_images_filenames)):
            filename=test_images_filenames[i]
            ima=cv2.imread(filename)
            gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
            kpt,des=Descriptor.extractKeyPointsAndDescriptors(gray)
            
            #Reduce the dimensionality of the data (PCA)
            new_des = pca.transform(des)
            
            #Predict the label for each descriptor
            predictions = clf.predict(stdSlr.transform(new_des))
            values, counts = np.unique(predictions, return_counts=True)
            predictedClasses[i] = values[np.argmax(counts)]
            print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedClasses[i]
    
        return predictedClasses