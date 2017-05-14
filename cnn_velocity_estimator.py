from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model
from scipy import misc
import json
import numpy as np

# Code should be put in a folder containing the benchmark_velocity folder
file_prefix = "./benchmark_velocity/clips/"

train_images_resized = []
train_velocities = []

# Use first half of data for training
for i in range(123):
    print i
    # Open annotation file
    annotation_file = file_prefix + "{0:0=3d}".format(i + 1) + "/annotation.json"
    with open(annotation_file) as json_data:
        annotations = json.load(json_data)
    # Extract bounding boxes and velocities of cars
    bboxes = []
    for j in range(len(annotations)):
        bbox = annotations[j]['bbox']
        bboxes.append((int(bbox['top']), int(bbox['bottom']), int(bbox['left']), int(bbox['right'])))
        train_velocities.append(annotations[j]['velocity'])
    image_folder = file_prefix + "{0:0=3d}".format(i + 1) + "/img/"
    images = []
    # Loop over bounding boxes for current example
    for bbox in bboxes:
        images_cropped = []
        # Loop over each frame of example
        for j in range(60):
            # Read in frame, crop it to bounding box
            image = image_folder + "{0:0=3d}".format(j + 1) +".jpg"
            image_data = misc.imread(image, mode = 'RGB')
            image_cropped = image_data[bbox[0]: bbox[1], bbox[2]:bbox[3]]
            images_cropped.append(image_cropped)
        # Concatenate cropped frames together into one image and resize it to 299 by 299 (default image size for Inception v3)
        images_merged = misc.imresize(np.concatenate(tuple(images_cropped), axis=1), (299, 299))
        # Append concatenation as example input for training
        train_images_resized.append(images_merged)    

train_images_resized = np.array(train_images_resized)
train_velocities = np.array(train_velocities)
np.save("train_images_merged_resized.npy", train_images_resized)
np.save("train_velocities.npy", train_velocities)
'''
train_images_resized = np.load("train_images_merged_resized.npy")
train_velocities = np.load("train_velocities.npy")
'''

# create the base pre-trained model (InceptionV3 pretrained on ImageNet)
# The base model and the fine training code was obtained from here:
# https://keras.io/applications/ 
# Changed loss to mean squared error and final prediction layer to output velocity
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and final prediction layer for velocity
predictions = Dense(2)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# train the model on the new data for a few epochs
model.fit(train_images_resized, train_velocities, epochs=10)

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
model.fit(train_images_resized, train_velocities, epochs=10)
model.save("cnn_velocity_estimator.h5")
print model.evaluate(train_images_resized, train_velocities)

test_images_resized = []
test_velocities = []
# Use second half of data for testing and preprocess it the same way as training data
for i in range(123):
    annotation_file = file_prefix + "{0:0=3d}".format(i + 124) + "/annotation.json"
    with open(annotation_file) as json_data:
        annotations = json.load(json_data)
    bboxes = []
    for j in range(len(annotations)):
        bbox = annotations[j]['bbox']
        bboxes.append((int(bbox['top']), int(bbox['bottom']), int(bbox['left']), int(bbox['right'])))
        test_velocities.append(annotations[j]['velocity'])
    image_folder = file_prefix + "{0:0=3d}".format(i + 124) + "/img/"
    images = []
    for bbox in bboxes:
        images_cropped = []
        for j in range(60):
            image = image_folder + "{0:0=3d}".format(j + 1) +".jpg"
            image_data = misc.imread(image, mode = 'RGB')
            image_cropped = image_data[bbox[0]: bbox[1], bbox[2]:bbox[3]]
            images_cropped.append(image_cropped)
        images_merged = misc.imresize(np.concatenate(tuple(images_cropped), axis=1), (299, 299))
        test_images_resized.append(images_merged)     

test_images_resized = np.array(test_images_resized)
test_velocities = np.array(test_velocities)
np.save("test_images_merged_resized.npy", test_images_resized)
np.save("test_velocities.npy", test_velocities)
print model.evaluate(test_images_resized, test_velocities)
'''
test_images_resized = np.load("test_images_merged_resized.npy")
test_velocities = np.load("test_velocities.npy")
model = load_model("cnn_velocity_estimator.h5")
'''
predicted_velocities = model.predict(test_images_resized)
np.save("predicted_velocities.npy", predicted_velocities)
print np.sum(np.square(test_velocities - predicted_velocities)) / (predicted_velocities.size + 0.0)
print np.sum(np.square(test_velocities - predicted_velocities)) / np.sum(np.square(test_velocities))

