from scipy import misc
import json
import numpy as np

'''
Implementation of recurrent CNN for predicting velocity. The network first computes
features by applying a CNN to every frame that is put in as input (currently last 5 frames 
of each sequence) and then uses these features as input into a recurrent neural network with 
1 LSTM hidden layer.

Inspired by this paper: 
Donahue et al. (2015), Long-term Recurrent Convolutional Networks for
Visual Recognition and Description

A warning: The training and test data take a lot of memory, which could lead to running out of 
RAM (unless your computer has a lot of RAM (like 16 GB or something like that). Also, at least for my 
computer, the network takes quite a long time to train (something like 10 minutes an epoch).
'''
# Code should be put in a folder containing the benchmark_velocity folder
file_prefix = "./benchmark_velocity/clips/"

'''
Below code is used to process a random half of the data as training data. 
Can modify it to add various amounts of padding to the bounding boxes since 
the bounding box of a car might change for different frames.
'''

train_sequences_resized = []
train_velocities = []

# Shuffle indices of data samples randomly (in case samples are ordered in some way)
random_indices = np.random.permutation(range(1, 247))
# Save random shuffling (useful later for extracting test set)
np.save("rcnn_shuffled_indices", random_indices)

# Use half of data for training
for i in range(123):
    print i
    # Open annotation file
    annotation_file = file_prefix + "{0:0=3d}".format(random_indices[i]) + "/annotation.json"
    with open(annotation_file) as json_data:
        annotations = json.load(json_data)
    # Extract bounding boxes and velocities of cars
    bboxes = []
    for j in range(len(annotations)):
        bbox = annotations[j]['bbox']
        # Add margin to bounding boxes
        # horiz_margin = 0.05 * (bbox['bottom'] - bbox['top'])
        # vert_margin =  0.05 * (bbox['right'] - bbox['left'])
        horiz_margin = 0
        vert_margin = 0
        bboxes.append((int(bbox['top'] - horiz_margin), int(bbox['bottom'] + horiz_margin), int(bbox['left'] - vert_margin), int(bbox['right'] +  vert_margin)))
        # Extract velocity
        train_velocities.append(annotations[j]['velocity'])
    image_folder = file_prefix + "{0:0=3d}".format(random_indices[i]) + "/img/"
    # Loop over bounding box for each car in sequence for current example
    for bbox in bboxes:
        # List of cropped frames in a sequence
        sequence_cropped = []
        # Loop over each frame of example
        for j in range(60):
            # Read in frame, crop it to bounding box, resize image to 299 by 299 (input image size to InceptionV3 CNN)
            image = image_folder + "{0:0=3d}".format(j + 1) +".jpg"
            image_data = misc.imread(image, mode = 'RGB')
            image_cropped = misc.imresize(image_data[bbox[0]: bbox[1], bbox[2]:bbox[3]], (299, 299))
            sequence_cropped.append(image_cropped)
        # Add sequence of cropped frames corresponding to a given car as an example sequence
        train_sequences_resized.append(images_cropped)    

train_sequences_resized = np.array(train_sequences_resized)
train_velocities = np.array(train_velocities)

np.save("rcnn_train_images_merged_resized_shuffle.npy", train_sequences_resized)
np.save("rcnn_train_velocities_shuffle.npy", train_velocities)

'''
# Load saved preprocessed training data (array of sequences of cropped frames corresponding to a car) and saved velocities
# Currently use last 5 frames of each sequence as input
train_sequences_resized = np.load("rcnn_train_images_merged_resized_shuffle.npy")[:, -5:, :, :]
train_velocities = np.load("rcnn_train_velocities_shuffle.npy")
'''

import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Input
from keras import backend as K
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

# set learning phase to 0 (this is necessary for the code to work due to batch normalization
# layers in the InceptionV3 network)
# See https://github.com/fchollet/keras/issues/5934 for more details.
K.set_learning_phase(0)

# create the base pre-trained model (InceptionV3 pretrained on ImageNet)
# The base model was obtained from here:
# https://keras.io/applications/.
input = Input(shape=(None, 299, 299, 3), name='input')
base_model = InceptionV3(weights='imagenet', include_top=False)

# Apply TimeDistributed wrapper to InceptionV3 model. This way, the same InceptionV3
# network is applied to every frame in a given input sequence. 
# Documentation for TimeDistributed wrapper: https://keras.io/layers/wrappers/
# I think in order to wrap the whole base_model with the TimeDistributed wrapper, a Lambda layer
# is needed (see https://gist.github.com/alfiya400/9d3bf303966f87a3c2aa92a0a0a54662)
cnn_time_model = TimeDistributed(Lambda(lambda x: base_model(x)))
cnn_time_output = cnn_time_model(input)
# Add a global spatial average pooling layer wrapped with TimeDistributed wrapper
cnn_time = TimeDistributed(GlobalAveragePooling2D())(cnn_time_output)
# Add a fully-connected layer again wrapped with TimeDistributed wrapper
rcnn_model = TimeDistributed(Dense(1024, activation='relu'))(cnn_time)
# Add LSTM hidden layer that takes as input features from the previous fully-connected
# layer wrapped with TimeDistributed wrapper
rcnn_model = LSTM(256)(rcnn_model)
# Add fully connected layer
rcnn_model = Dense(128, activation='relu')(rcnn_model)
# Output layer that predicts both components of velocity
rcnn_model = Dense(2)(rcnn_model)

# this is the model we will train
model = Model(inputs=[input], outputs=rcnn_model)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# train the model on the new data for a few epochs
model.fit(train_sequences_resized, train_velocities, epochs=10)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mean_squared_error')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(train_sequences_resized, train_velocities, epochs=10)

# Save weights of model (due to the Lambda layer, saving 
# the whole model with model.save and loading model with load_model doesn't work)
model.save_weights("rcnn_velocity_estimator_shuffled.h5")

# Predict velocities for training data and evaluate model on training data
train_predicted_velocities = model.predict(train_sequences_resized)
np.save("rcnn_train_predicted_velocities_shuffle.npy", train_predicted_velocities)
print model.evaluate(train_sequences_resized, train_velocities)
print np.sum(np.square(train_velocities - train_predicted_velocities)) / np.sum(np.square(train_velocities))
'''
Below code (currently commented out) can be used to process other half of the data 
as test data. Can modify it to add various amounts of padding to the bounding boxes since 
the bounding box of a car might change for different frames.
'''

test_sequences_resized = []
test_velocities = []

# Use other half of data for testing
for i in range(123, 246):
    print i
    # Open annotation file
    annotation_file = file_prefix + "{0:0=3d}".format(random_indices[i]) + "/annotation.json"
    with open(annotation_file) as json_data:
        annotations = json.load(json_data)
    # Extract bounding boxes and velocities of cars
    bboxes = []
    for j in range(len(annotations)):
        bbox = annotations[j]['bbox']
        # horiz_margin = 0.05 * (bbox['bottom'] - bbox['top'])
        # vert_margin =  0.05 * (bbox['right'] - bbox['left'])
        horiz_margin = 0
        vert_margin = 0
        bboxes.append((int(bbox['top'] - horiz_margin), int(bbox['bottom'] + horiz_margin), int(bbox['left'] - vert_margin), int(bbox['right'] +  vert_margin)))
        test_velocities.append(annotations[j]['velocity'])
    image_folder = file_prefix + "{0:0=3d}".format(random_indices[i]) + "/img/"
    images = []
    # Loop over bounding boxes for current example
    for bbox in bboxes:
        sequence_cropped = []
        # Loop over each frame of example
        for j in range(60):
            # Read in frame, crop it to bounding box
            image = image_folder + "{0:0=3d}".format(j + 1) +".jpg"
            image_data = misc.imread(image, mode = 'RGB')
            image_cropped = misc.imresize(image_data[bbox[0]: bbox[1], bbox[2]:bbox[3]], (299, 299))
            sequence_cropped.append(image_cropped)
        test_sequences_resized.append(images_cropped)    

test_sequences_resized = np.array(test_sequences_resized)
test_velocities = np.array(test_velocities) 
np.save("rcnn_test_images_merged_resized_shuffle.npy", test_sequences_resized)
np.save("rcnn_test_velocities_shuffle.npy", test_velocities)
'''
# Load saved preprocessed test data (array of sequences of cropped frames corresponding to a car) and saved velocities
# Currently use last 5 frames of sequence as input
test_sequences_resized = np.load("rcnn_test_images_merged_resized_shuffle.npy")[:, -5:, :, :]
test_velocities = np.load("rcnn_test_velocities_shuffle.npy")
'''
# Predict velocities for test data and evaluate model on test data
print model.evaluate(test_sequences_resized, test_velocities)
predicted_velocities = model.predict(test_sequences_resized)
np.save("rcnn_predicted_velocities_shuffle.npy", predicted_velocities)

print np.sum(np.square(test_velocities - predicted_velocities)) / np.sum(np.square(test_velocities))
