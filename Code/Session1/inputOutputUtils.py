import cPickle

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