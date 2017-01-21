
import numpy as np
from multiprocessing import Pool
import sys

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

def getBaseModel():
    CNN_base_model = VGG16(weights='imagenet')
    return CNN_base_model

   
def useAverage(des, output_descriptors):
    
        
    return des


def getDescriptors(x, layer_taken, CNN_base_model, CNN_new_model, method_used):
    #Descriptors depending on the chosen layer
    if layer_taken == 'fc2' or layer_taken == 'fc1':
        #Extract features from last layer
        des=CNN_new_model.predict(x)
    elif layer_taken == 'block5_pool':
        clear_zeros = method_used['clear_zeros']
        #From block5_pool layer we get a matrix of dim (512, 7, 7)
        features = CNN_new_model.predict(x)
        features = np.squeeze(features, axis = 0)
        num_des = features.shape[0]
        length_des = features.shape[1]*features.shape[2]
        if clear_zeros:
            des = np.reshape(features[0, :, :], (1, length_des))
    
            for i in range(1, num_des):
                feat = np.reshape(features[i, :, :], (1, length_des))
                if sum(feat) != 0: 
                    des = np.vstack(des, feat)
        else:    
            des = np.zeros([num_des, length_des])
    
            for i in range(num_des):
                des[i, :] = np.reshape(features[i, :, :], (1, length_des))
        if method_used['method_to_reduce_dim'] != 'Nothing':   
            method = getattr(sys.modules[__name__], 'use' + method_used['method_to_reduce_dim'])
            des = method(des, method_used['Remaining_features'])
        #TO DO: decide how to get the information from this layer
    #des must be a numpy array with rows corresponding to different descriptors
    return des

#Extract features methods
def extractFeaturesMaps(FLSubset, layer_taken, CNN_base_model, num_slots, method_used):
    #Crop the model up to a certain layer
    CNN_new_model = Model(input=CNN_base_model.input, output=CNN_base_model.get_layer(layer_taken).output)
    
    #Shared data for pool
    data = [layer_taken, CNN_base_model, CNN_new_model, method_used]
    
    pool = Pool(processes = num_slots, initializer = initPool, initargs = [data])
    deslab = pool.map(getFeaturesAndLabelsForImage, FLSubset)
    pool.terminate()

    labels = [x[1] for x in deslab]
    Train_descriptors = [x[0] for x in deslab]
    
    # Transform everything to numpy arrays
    Train_label_per_descriptor = labels
    
    if layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten':
        #Not BoVW
        D=Train_descriptors[0]
        for i in range(1,len(Train_descriptors)):
            D=np.vstack((D,Train_descriptors[i]))
    else:
        
        size_descriptors = Train_descriptors[0].shape[1]
        
        D = np.zeros((np.sum([p.shape[0] for p in Train_descriptors]), size_descriptors), dtype=np.float32)

        startingpoint = 0
                
        for i in range(len(Train_descriptors)): 
            D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
            startingpoint += len(Train_descriptors[i])
            
        
    return D, Train_descriptors, Train_label_per_descriptor

def getFeaturesAndLabelsForImage((filename,label)):
    print 'Reading image ' + filename + ' with label ' + label
    layer_taken = data[0]
    CNN_base_model = data[1]
    CNN_new_model = data[2]
    method_used = data[3]
    
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)

    des = getDescriptors(x, layer_taken, CNN_base_model, CNN_new_model, method_used)

    return (des, label)

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_