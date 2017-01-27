from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input

img_width=224
img_height=224

def getDataGenerator():
    #datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,preprocessing_function=preprocess_input, rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,fill_mode='nearest',horizontal_flip=False,vertical_flip=False,rescale=None)
    datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False, rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,fill_mode='nearest',horizontal_flip=False,vertical_flip=False,rescale=None)
    
    return datagen

def getAugmentedDataGenerator():
    #datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,preprocessing_function=preprocess_input, rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,fill_mode='nearest',horizontal_flip=False,vertical_flip=False,rescale=None)
    datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False, rotation_range=0.,width_shift_range=0.,height_shift_range=0.,shear_range=0.,zoom_range=0.,fill_mode='nearest',horizontal_flip=False,vertical_flip=False,rescale=None)
    
    #TODO: Data augmentation
    
    return datagen

def getData(datagen, augmented_datagen, batch_size):
    train_data_dir='./Databases/MIT/train'
    test_data_dir='./Databases/MIT/validation'
    val_data_dir='./Databases/MIT/test'
    
    train_generator = augmented_datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
    val_generator = datagen.flow_from_directory(val_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
    test_generator = datagen.flow_from_directory(test_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical')
    
    return train_generator,val_generator,test_generator