import cPickle
import numpy as np

filenames_train_file='train_images_filenames.dat'
filenames_test_file='test_images_filenames.dat'
labels_train_file='train_labels.dat'
labels_test_file='test_labels.dat'

def readData():
    train_images_filenames = cPickle.load(open(filenames_train_file,'r'))
    test_images_filenames = cPickle.load(open(filenames_test_file,'r'))
    train_labels = cPickle.load(open(labels_train_file,'r'))
    test_labels = cPickle.load(open(labels_test_file,'r'))
    
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

    return train_images_filenames,test_images_filenames,train_labels,test_labels
    
def getFilenamesLabelsSubset(train_images_filenames,train_labels,max_class_train_images):
    uniquelabels=[label for label in np.unique(train_labels)]
    FLSubset=[]
    for label in uniquelabels:
        idx=train_labels.index(label)
        train_images_subset=train_images_filenames[idx:idx+max_class_train_images]
        train_labels_subset=train_labels[idx:idx+max_class_train_images]
        FLSubset[len(FLSubset):] = zip(train_images_subset,train_labels_subset)
    return FLSubset