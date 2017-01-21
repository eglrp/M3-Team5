import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
import descriptors,BoW

from keras.models import Model

def trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced'):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight).fit(D_scaled, Train_label_per_descriptor)
    print 'Done!'

    return clf,stdSlr

def predictBoVW(Split, layer_taken, stdSlr, codebook, k, CNN_base_model, num_slots, pca, method_used):
    #Compute features
    D, Train_descriptors, Train_label_per_descriptor = descriptors.extractFeaturesMaps(Split, layer_taken, CNN_base_model, num_slots, method_used)
    
    if pca != None:
        D = pca.transform(D)
        
        for idx,TrainDes in enumerate(Train_descriptors):        
            train_descriptor = pca.transform(TrainDes)
            Train_descriptors[idx]=train_descriptor
                             
    #Determine visual words
    visual_words_test = BoW.getVisualWords(codebook, k, Train_descriptors)
    #Apply PCA
    predictedLabels = stdSlr.transform(visual_words_test)
    
    return predictedLabels

def predict(Split, layer_taken, stdSlr, clf, CNN_base_model, num_slots):
    #Compute features
    CNN_new_model = Model(input=CNN_base_model.input, output=CNN_base_model.get_layer(layer_taken).output)
    
    data = [layer_taken,clf,stdSlr,CNN_base_model,CNN_new_model]#shared data with processes
    
    pool = Pool(processes=4,initializer=initPool, initargs=[data])
    predictedClasses= pool.map(getPredictionForImage, Split)
    pool.terminate()
    
    predictions=[str(x) for x in predictedClasses]
    
    return predictions

def getPredictionForImage((filename,label)):
    layer_taken=data[0]
    computedClf=data[1]
    computedstdSlr=data[2]
    CNN_base_model=data[3]
    CNN_new_model=data[4]
    
    descriptors.initPool([layer_taken,CNN_base_model,CNN_new_model])
    deslab=descriptors.getFeaturesAndLabelsForImage((filename,label))
    
    #Predict label
    predictions = computedClf.predict(computedstdSlr.transform(deslab[0]))
    values, counts = np.unique(predictions, return_counts=True)
    predictedClass = values[np.argmax(counts)]
    
    return predictedClass

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_