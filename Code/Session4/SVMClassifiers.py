import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from multiprocessing import Pool
import descriptors

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

def trainSVM(visual_words,Train_label_per_descriptor,Cparam=1,kernel_type='linear',degree_value=1,gamma_value=0.01,weight = 'balanced'):
    # Train a SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernel_type, C=Cparam,degree=degree_value,gamma=gamma_value,class_weight=weight).fit(D_scaled, Train_label_per_descriptor)
    print 'Done!'

    return clf,stdSlr

def predict(test_images_filenames, layer_taken, stdSlr, codebook, k, num_slots):
    #Predict test set labels with the trained classifier
    CNN_base_model = VGG16(weights='imagenet')
    data = [codebook, k, layer_taken, CNN_base_model]#shared data with processes
    
    pool = Pool(processes=num_slots,initializer=initPool, initargs=[data])
    
    visual_words_test = pool.map(getVisualWordsForImage, test_images_filenames)
    if not codebook:
        vw = np.zeros([len(visual_words_test), len(visual_words_test[0][0])], dtype = np.float32)
        for i in range(len(visual_words_test)):
            vw[i, :] = visual_words_test[i][0]
        visual_words_test = vw
        
    pool.terminate()

    predictedLabels = stdSlr.transform(visual_words_test)
    
    return predictedLabels

def getVisualWordsForImage(filename):
    
    codebook = data[0]
    k = data[1]
    layer_taken = data[2]
    CNN_base_model = data[3]
    
    #Prerocess image
    img = image.load_img(filename, target_size = (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    im = preprocess_input(x)    
    #Obtain descriptors from CNN layer
    des = descriptors.getDescriptors(im, layer_taken, CNN_base_model)
    
    if not(layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten'):
        #Predict the label for each descriptor, when necessary
        words = codebook.predict(des)
        visual_words = np.bincount(words, minlength = k)
    else:
        visual_words = des
        
    return visual_words

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_