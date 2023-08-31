#%% Image Segmentation: Extract from the rest of the image
#We already did it as follwoing
#2.2 image_opencv_arithmetic.py
#2.3 image_opencv_contours.py
#2.4 image_opencv_filtering.py
#3. image_opencv_face_detection.py
#4. image_opencv_feature_detection.py
#6. image_tensorflow_object_detection.py
#7. image_digit_recognition.py

#%% Common to all
import os
os.chdir("D:\\trainings\\computer_vision")

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import the necessary packages specific to Computer vision
import cv2

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Color Quantization: reduce number of colors in an image OR group colurs in few clusters

# Read and process
my_image_color = cv2.imread('./image_data/balloon.jpg')
show_image([my_image_color], row_plot = 1)

#To do KMean, convert/reshape the image to 'number of pixels' x BGR
my_image_color_BGR = np.float32(my_image_color.reshape((-1,3)))
my_image_color_BGR
my_image_color_BGR.shape # 300*225 -> my_image_color.shape

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
clusters_count = 3
ret, clusters_label, clusters_centers = cv2.kmeans(my_image_color_BGR, clusters_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8
clusters_centers = np.uint8(clusters_centers)
#Row side is for cluster 1,2,3

#Make original image with center color
my_image_clusters_color = clusters_centers[clusters_label.flatten()]

#reshape it back to the shape of original image
my_image_clusters_color = my_image_clusters_color.reshape((my_image_color.shape))
show_image([my_image_color, my_image_clusters_color], row_plot = 1)

#CW: Take different image and practice