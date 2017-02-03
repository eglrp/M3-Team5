from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

img_width = 256
img_height = 256

def getDataGenerator():
    datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,preprocessing_function=preprocess_input, rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,fill_mode='nearest',horizontal_flip=False,vertical_flip=False,rescale=None)
    
    return datagen

def getAugmentedDataGenerator():
    augmented_datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,preprocessing_function=preprocess_input, rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,fill_mode='nearest',horizontal_flip=True,vertical_flip=False,rescale=None,zca_whitening=True)
    
    #TODO: Data augmentation
    
    return augmented_datagen

def getDataOld(datagen, augmented_datagen, batch_size):
    train_data_dir='./Databases/MIT/train'
    test_data_dir='./Databases/MIT/validation'
    val_data_dir='./Databases/MIT/test'
    
#    train_generator = augmented_datagen.flow_from_directory(train_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
    train_generator = datagen.flow_from_directory(train_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
    val_generator = datagen.flow_from_directory(val_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
    test_generator = datagen.flow_from_directory(test_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
    
    return train_generator,val_generator,test_generator

def getData(datagen, batch_size):
    train_data_dir = './Databases/MIT/train'
    test_data_dir = './Databases/MIT/validation'
    val_data_dir = './Databases/MIT/test'
    
    train_generator = datagen.flow_from_directory(train_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
    val_generator = datagen.flow_from_directory(val_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = 'categorical')
    test_generator = datagen.flow_from_directory(test_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
    
    return train_generator, val_generator, test_generator

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x