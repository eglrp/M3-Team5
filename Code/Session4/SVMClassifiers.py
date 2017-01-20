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

def predict(test_images_filenames,stdSlr, codebook,k, num_slots):
    #Predict test set labels with the trained classifier
    data = [codebook,k]#shared data with processes
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    
    visual_words_test=np.zeros((len(test_images_filenames),k), dtype=np.float32)
    visual_words_test= pool.map(getVisualWordsForImage, test_images_filenames)

    pool.terminate()

    predictedLabels = stdSlr.transform(visual_words_test)
    
    return predictedLabels

def getVisualWordsForImage(filename):
    codebook=data[0]
    k=data[1]
    
    
    #CNN
    
    

    #Predict the label for each descriptor
    words=codebook.predict(des)
    visual_words=np.bincount(words,minlength=k)
    
    return visual_words

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_