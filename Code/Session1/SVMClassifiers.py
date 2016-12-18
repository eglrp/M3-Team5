import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
import descriptors

def trainSVM(D,L,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced'):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(D)
    D_scaled = stdSlr.transform(D)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight).fit(D_scaled, L)
    print 'Done!'

    return clf,stdSlr

def predict(test_images_filenames,descriptor_type,clf,stdSlr, pca):
    #Predict test set labels with the trained classifier
    data = [pca,clf,stdSlr,descriptor_type]#shared data with processes
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    predictedClasses= pool.map(getPredictionForImage, test_images_filenames)
    pool.terminate()
    
    predictions=[str(x) for x in predictedClasses]
    
    return predictions

def getPredictionForImage(filename):
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