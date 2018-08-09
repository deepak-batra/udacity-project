import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'capstone_images/train'
validation_data_dir = 'capstone_images/valid'
test_data_dir = 'capstone_images/test'
nb_train_samples = 6680
nb_validation_samples = 1597
nb_test_samples = 3566
epochs = 50
batch_size = 20

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the RESNET50 network
    model = applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    # np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    # np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_test = model.predict_generator(
        generator, nb_test_samples // batch_size)
    # np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)

    np.savez(open('CapstoneResnet50Data.npz', 'wb+'), train=bottleneck_features_train, valid=bottleneck_features_validation, test=bottleneck_features_test)

save_bottlebeck_features()            
