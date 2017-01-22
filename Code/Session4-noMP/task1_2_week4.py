# -*- coding: utf-8 -*-
"""

@author: master
nv"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


applyMean = False
applyPca = False
visualizingWeights = False

#load VGG model
base_model = VGG16(weights='imagenet')
#visalize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

#read and process image
img_path = '../../Databases/MIT_split/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
plt.imshow(img)
plt.show()

#crop the model up to a certain layer
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)
#get the features from images
features = model.predict(x)

#crop the model up to a certain layer (layer 3)
model3 = Model(input=base_model.input, output=base_model.get_layer('block3_conv2').output)
#get the features from images
newfeatures = model3.predict(x)

if applyPca:
    
    vector= np.reshape(newfeatures[0],[newfeatures.shape[1],newfeatures.shape[2]*newfeatures.shape[3]])
    transpvector = vector.T
    pca = PCA(n_components = 1)
    pca.fit(transpvector)
    newD=pca.transform(transpvector)
    img = np.reshape(newD,[newfeatures.shape[2],newfeatures.shape[3]])
    plt.imshow(img)

if applyMean:

    avg = np.mean(newfeatures[0],axis=0)
    plt.imshow(avg)

#    aux = np.zeros([len(newfeatures[0][0]),len(newfeatures[0][0][0])])
#    for x in range(len(newfeatures[0])):
#        smm = aux + newfeatures[0][x]
#    aux = smm/len(newfeatures[0])
#    plt.imshow(aux)

weights = base_model.get_layer('block1_conv1').get_weights()
#Visualizing weights
if visualizingWeights:
    pesos = np.zeros([24,24,3])
    for dim in range(0,weights[0].shape[3]):
        counter = 0
        for i in range(pesos.shape[0]/3):
            for j in range(pesos.shape[1]/3):
                pesos[3*i:3*i+3,3*j:3*j+3,dim] = weights[0][counter,dim,:,:]
                counter = counter + 1;
    #avg = np.mean(pesos,axis=2)
    #pesos = weights[0].reshape((24,24,3))

    plt.imshow(pesos,interpolation = 'nearest')
    