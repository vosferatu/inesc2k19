import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.models import load_model
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_labels = []
train_samples = []
test_labels= []
test_samples = []

##################
# GENARATED DATA #
##################
for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(10):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

#print raw data
#for i in train_samples:
 #   print(i)

#converting list into np array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

test_labels = np.array(train_labels)
test_samples = np.array(train_samples)

# Transformation so the neural network can train the data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))
scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))
#for i in scaled_train_samples:
 #   print(i)


#########################
# CREATE AN NN IN KERAS #
#########################

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
    ])

#model.summary()

model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

#for i in predictions:
 #   print(i)

rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)

#for i in rounded_predictions:
 #   print(i)

####################
# CONFUSION MATRIX #
####################

cm = confusion_matrix(test_labels, rounded_predictions)

def plot_confusion_matrix(cm, classes, normalize=False, title = 'Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment="center", color="white" if cm[i, j]>thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

cm_plot_labels = ["no_side_effects", "had_side_effects"]
#plot_confusion_matrix(cm, cm_plot_labels, title="confusion Matrix")

##############
# SAVE MODEL #
##############

model.save('medical_trial_model.h5')
new_model = load_model('medical_trial_model.h5')

new_model.summary()

##############
# SAVE MODEL #
##############

