
import numpy as np
from multiprocessing import Pool

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model



def getDescriptors(im, layer_taken, CNN_base_model):
    #Crop the model up to a certain layer
    CNN_new_model = Model(input = CNN_base_model.input, output = CNN_base_model.get_layer(layer_taken).output)
    
    #Decide what to do depending on the last layer chosen
    if layer_taken == 'fc2':
        des=0
        #TO DO: decide how to get the information from this layer
    elif layer_taken == 'fc1':
        des=0
        #TO DO: decide how to get the information from this layer
    elif layer_taken == 'block5_pool':
        des=0
        
        #TO DO: decide how to get the information from this layer
    #des must be a numpy array with rows corresponding to different descriptors
    return des

#Extract features methods
def extractFeatures(FLSubset, layer_taken, num_slots):
    CNN_base_model = VGG16(weights='imagenet')
    
    #Common data for pool
    data = [layer_taken, CNN_base_model]
    
    pool = Pool(processes = num_slots, initializer = initPool, initargs = [data])
    deslab = pool.map(getDescriptorsAndLabelsForImage, FLSubset)
    pool.terminate()

    Train_label_per_descriptor = [x[1] for x in deslab]
    Train_descriptors = [x[0] for x in deslab]
    
    # Transform everything to numpy arrays
    size_descriptors = Train_descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.float32)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
        startingpoint += len(Train_descriptors[i])
    
    return D, Train_descriptors, Train_label_per_descriptor

def getDescriptorsAndLabelsForImage((filename,label)):
    print 'Reading image ' + filename + ' with label ' + label
    layer_taken = data[0]
    CNN_base_model = data[1]
    
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    im = preprocess_input(x)    

    des = getDescriptors(im, layer_taken, CNN_base_model)

    return (des, label)

#Multiprocessing utils
def initPool(data_):
    # data to share with processes
    global data
    data = data_


    
