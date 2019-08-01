from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

image_width = 768
image_height = 576

train_dir = './CNN/images/train'
validation_dir = './CNN/images/valid'

# Load the VGG model
conv_base = VGG16(weights='imagenet', include_top=False,
                  input_shape=(image_width, image_height, 3))


base_dir = '/home/vosferatu/Desktop/inesc2k19/CNN/images'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 16


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 24, 18, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 40)
validation_features, validation_labels = extract_features(validation_dir, 40)
test_features, test_labels = extract_features(test_dir, 10)

train_features = np.reshape(train_features, (40, 24 * 18 * 512))
validation_features = np.reshape(validation_features, (10, 24 * 18 * 512))
test_features = np.reshape(test_features, (10, 24 * 18 * 512))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=24 * 18 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=10,
                    batch_size=16,
                    validation_data=(validation_features, validation_labels))


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

mdl = models.Sequential()
mdl.add(conv_base)
mdl.add(layers.Flatten())
mdl.add(layers.Dense(256, activation='relu'))
mdl.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False
