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
    
def getTrainingValidationSplit(train_images_filenames,train_labels,train_percentage):
    uniquelabels,numlabels=[label for label in np.unique(train_labels, return_counts=True)]
    TrainingSplit=[]
    ValidationSplit=[]
    i=0
    for label in uniquelabels:
        max_class_train_images=int(np.round(numlabels[i]*train_percentage))
        idx=train_labels.index(label)
        train_images_subset=train_images_filenames[idx:idx+max_class_train_images]
        train_labels_subset=train_labels[idx:idx+max_class_train_images]
        TrainingSplit[len(TrainingSplit):] = zip(train_images_subset,train_labels_subset)
        
        validation_images=train_images_filenames[idx+max_class_train_images+1:idx+numlabels[i]]
        validation_labels=train_labels[idx+max_class_train_images+1:idx+numlabels[i]]
        ValidationSplit[len(ValidationSplit):] = zip(validation_images,validation_labels)
        i=i+1
    return TrainingSplit, ValidationSplit
    
def unzipTupleList(tupleList):
    first,second=zip(*tupleList)
    first=list(first)
    second=list(second)
    return first,second