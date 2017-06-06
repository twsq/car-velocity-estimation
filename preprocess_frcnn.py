import pickle
import json

'''
This code creates the .txt file that should be fed in for the Keras Faster R-CNN 
implementation.
'''
# Code should be put in a folder containing the benchmark_velocity_supp folder
file_prefix = "/data/home/ubuntu/benchmark_velocity_supp/"

train_proportion = 1
num_supp_images = 5067

# Open annotation file containing ground truth bounding boxes
annotation_file = file_prefix + "annotation.json"
with open(annotation_file) as json_data:
    annotations = json.load(json_data)

processed_data = open("image_bboxes_train.txt", "w+")
# Preprocess data and write to .txt file
for i in range(int(num_supp_images * train_proportion)):
    print i
    image_annotations = annotations[i]
    image_file_name = image_annotations["file_name"]
    image_bboxes = image_annotations["bbox"]
    full_file_name = (file_prefix + image_file_name).encode('utf8')
    for bbox in image_bboxes:
        processed_data.write(full_file_name)
        processed_data.write(',')
        processed_data.write(str(int(bbox["left"])))
        processed_data.write(',')
        processed_data.write(str(int(bbox["top"])))
        processed_data.write(',')
        processed_data.write(str(int(bbox["right"])))
        processed_data.write(',')
        processed_data.write(str(int(bbox["bottom"])))
        processed_data.write(',')
        processed_data.write("car")
        processed_data.write("\n")
processed_data.close()

