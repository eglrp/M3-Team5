
import numpy as np
from multiprocessing import Pool

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

def getBaseModel():
    CNN_base_model = VGG16(weights='imagenet')
    return CNN_base_model

def getDescriptors(x, layer_taken, CNN_base_model, CNN_new_model):
    #Descriptors depending on the chosen layer
    if layer_taken == 'fc2' | layer_taken == 'fc1':
        #Extract features from last layer
        des=CNN_new_model.predict(x)
    elif layer_taken == 'block5_pool':
        des = np.array([0, 0])
        
        #TO DO: decide how to get the information from this layer
    #des must be a numpy array with rows corresponding to different descriptors
    return des

#Extract features methods
def extractFeaturesMaps(FLSubset, layer_taken, CNN_base_model, num_slots):
    #Crop the model up to a certain layer
    CNN_new_model = Model(input=CNN_base_model.input, output=CNN_base_model.get_layer(layer_taken).output)
    
    #Shared data for pool
    data = [layer_taken, CNN_base_model, CNN_new_model]
    
    pool = Pool(processes = num_slots, initializer = initPool, initargs = [data])
    deslab = pool.map(getFeaturesAndLabelsForImage, FLSubset)
    pool.terminate()

    labels = [x[1] for x in deslab]
    Train_descriptors = [x[0] for x in deslab]
    #print Train_descriptors[0]
    #print len(Train_descriptors[0])
    #print type(Train_descriptors)
    #print type(Train_descriptors[0])
    # Transform everything to numpy arrays
    
    if layer_taken == 'fc1' or layer_taken == 'fc2' or layer_taken == 'flatten':
        #Not BoVW
        Train_label_per_descriptor=labels
        D=Train_descriptors[0]
        for i in range(1,len(Train_descriptors)):
            D=np.vstack((D,Train_descriptors[i]))
    else:
        size_descriptors = Train_descriptors[0].shape[1]
        D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.float32)
        #print D.shape
        
        startingpoint = 0
        Train_label_per_descriptor = np.array([labels[0]]*Train_descriptors[0].shape[0])
        D[startingpoint:startingpoint + len(Train_descriptors[0])] = Train_descriptors[0]
        startingpoint += len(Train_descriptors[0])
        
        for i in range(1,len(Train_descriptors)):
            Train_label_per_descriptor = np.hstack((Train_label_per_descriptor,np.array([labels[i]]*Train_descriptors[i].shape[0])))
            D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
            startingpoint += len(Train_descriptors[i])
    
    return D, Train_descriptors, Train_label_per_descriptor

def getFeaturesAndLabelsForImage((filename,label)):
    print 'Reading image ' + filename + ' with label ' + label
    layer_taken = data[0]
    CNN_base_model = data[1]
    CNN_new_model = data[2]
    
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)

    des = getDescriptors(x, layer_taken, CNN_base_model, CNN_new_model)

    return (des, label)

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_