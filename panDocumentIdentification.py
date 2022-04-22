import os
import pathlib
import tensorflow as tf
import cv2
import argparse
import string
import random
import numpy as np
from PIL import Image
import warnings
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from skew import correct_skew
from threshADP import adpThresh
from panOCR import ocrPan

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.gfile = tf.io.gfile

def randomStringGenerator():
    randomStr = ''.join(random.choices(string.ascii_lowercase, k = 10))
    return randomStr

PATH_TO_MODEL_DIR = 'my_model_pan'

PATH_TO_LABELS = 'annotations_pan/label_map.pbtxt'

MIN_CONF_THRESH = float(0.60)

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=False)

pan_keywords = ['आयकर', 'विभाग','income', 'tax', 'department','भारत', 'permanent', 'account', 'number','signature','permanent', 'account', 'number', 'card','स्थायी', 'खाता', 'संख्या', 'कार्ड','हस्ताक्षर']

def convertTensors(image):
    input_tensor = tf.convert_to_tensor(image)
    return input_tensor

def tensorAxis(ip_tensor):
    input_tensor = ip_tensor[tf.newaxis, ...]
    return input_tensor

def detection(ip_tensor):
    detections = detect_fn(ip_tensor)
    return detections

def numDetection(detection):
    num_detection = int(detection.pop('num_detections'))
    detection = {key: value[0, :num_detection].numpy()
                 for key, value in detection.items()}
    detection['num_detections'] = num_detection
    detection['detection_classes'] = detection['detection_classes'].astype(
        np.int64)
    return detection

def confidence(detection):
    conf = 0
    for scores in detection['detection_scores']:
        if scores > 0.5:
            conf = conf + scores
    conf = conf*10
    return conf

def upload_image_pan_doc(filestream):
    image = filestream
    single_input_tensor = convertTensors(image)
    input_tensor = tensorAxis(single_input_tensor)
    detections = detection(input_tensor)
    detections = numDetection(detections)
    image_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
        agnostic_mode=False)

    conf = confidence(detections)
    if(conf > 0.5):
        is_pan = True
    else:
        is_pan = False

    randomStrRotated = randomStringGenerator()
    randomStrThresh = randomStringGenerator()

    angle, skewed = correct_skew(image)
    cv2.imwrite(randomStrRotated + "Skewed.jpg", skewed)

    skewedImageName = randomStrRotated + "Skewed.jpg"

    gray_image = cv2.imread(skewedImageName, cv2.IMREAD_GRAYSCALE)
    thresh_adp_image = adpThresh(gray_image)
    cv2.imwrite(randomStrThresh + "Thresh.jpg", thresh_adp_image)

    threshImageName = randomStrThresh + "Thresh.jpg"            
    
    ocr = ocrPan(threshImageName)

    try: 
        os.remove(skewedImageName)
        os.remove(threshImageName)
    except: pass
    counter = 0
    if is_pan == True:
        for keyword in pan_keywords:
            if keyword in ocr:
                counter = counter + 1
    
    if counter > 1:
        is_pan = True
    else:
        is_pan = False

    result = {
        "Pan Card Found": is_pan,
        "Confidence": conf
    }
    return result


if __name__ == "__main__":
    image_path = "test_images_pan/3.jpg"
    npimg = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    res = upload_image_pan_doc(image)
    print(res)
    print("Done!!!")
