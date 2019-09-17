######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = 'C:/tensorflow1/models/research/object_detection/exported_graphs/frozen_inference_graph.pb'
#PATH_TO_CKPT = 'C:/tensorflow1/models/research/object_detection/second_model_inference/frozen_inference_graph.pb'


# Path to label map file
PATH_TO_LABELS = 'C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt'
# Path to image
PATH_TO_IMAGE = "C:/tensorflow1/models/research/object_detection/test_images/Scanning Image/Validation Set/"

PATH_TO_RESULTS = "C:/tensorflow1/models/research/object_detection/test_images/Scanning Image/Validation Set/Results/Full Image Scan/"

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

VALIDATION_IMAGES_FOLDERS = os.listdir(PATH_TO_IMAGE)
bounding_boxes = []
for folder in VALIDATION_IMAGES_FOLDERS:
    #if folder == "Results":
     #   continue

    if not folder == "4 images":
        continue
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    images = os.listdir(PATH_TO_IMAGE + folder + "/")
    print ("Current directory: " + folder)
    for image_name in images:
        image_path = PATH_TO_IMAGE + folder + "/" +  image_name
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        # All the results have been drawn on image. Now display the image.
        cv2.imshow("Full Imahe Scan", image)
        cv2.imwrite(PATH_TO_RESULTS +folder + "/" + image_name ,image)
        print("Saved " + PATH_TO_RESULTS +folder + "/" + image_name)
        
        min_score_thresh = 0.6 
        bboxes = boxes[scores > min_score_thresh]

        #get image size
        im_width, im_height = (640,640)
        final_box = []
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            new_value = [xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]
            final_box.append(new_value)
            bounding_boxes.append([image_name, new_value[0],  new_value[1],  new_value[2],  new_value[3]])
            #bounding_boxes.update({image_name : new_value})
            
         
        
        print (final_box)
        

    # Press any key to close the image
    #cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()



with open('bounding_boxes.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    for box in bounding_boxes :
        writer.writerow([box[0], box[1], box[2], box[3], box[4]])

csvFile.close()
       
