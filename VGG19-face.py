from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

base_model = VGG19(weights='imagenet')
#model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
# add a global spatial average pooling layer
x = base_model.output
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
base_model.compile(optimizer='adam', loss='categorical_crossentropy')

base_model.summary()

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'CNN/images/train',
        target_size=(224, 224),
        batch_size=8,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'CNN/images/valid',
        target_size=(224, 224),
        batch_size=8,
        class_mode='binary')

callbacks = [ModelCheckpoint('modelo29julho',save_best_only=True)]

model.fit_generator(
        train_generator,
        steps_per_epoch=5,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=1,
        callbacks=callbacks)

model.save('modelo29dejulho.h5')
