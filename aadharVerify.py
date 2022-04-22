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
import json
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from flask import Flask, jsonify, request, redirect, render_template

from skew import correct_skew
from threshADP import adpThresh
from ocr import ocrData

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.gfile = tf.io.gfile


def randomStringGenerator():
    randomStr = ''.join(random.choices(string.ascii_lowercase, k=10))
    return randomStr


PATH_TO_MODEL_DIR = 'my_model_aadhar'

PATH_TO_LABELS = 'annotations_aadhar/label_map.pbtxt'

MIN_CONF_THRESH = float(0.60)

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=False)


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


def detectionScoreCalculation(detection):
    detect_class = detection['detection_classes']
    detect_score = detection['detection_scores']
    detection_class = []
    for cls in detect_class[:4]:
        if cls == 1:
            detection_class.append("Amblem-logo")
        elif cls == 2:
            detection_class.append("Government Of India -logo")
        elif cls == 3:
            detection_class.append("Aadhaar-logo")
        elif cls == 4:
            detection_class.append("Aadhaar-tagline")

    detection_score = []
    for score in detect_score[:4]:
        detection_score.append(score*100)

    calculated_score = dict(zip(detection_class, detection_score))
    return calculated_score


def upload_image_aadhar(filestream):
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
        is_aadhar = True
    else:
        is_aadhar = False

    randomStrRotated = randomStringGenerator()
    randomStrThresh = randomStringGenerator()

    angle, skewed = correct_skew(image)
    cv2.imwrite(randomStrRotated + "Skewed.jpg", skewed)

    skewedImageName = randomStrRotated + "Skewed.jpg"

    gray_image = cv2.imread(skewedImageName, cv2.IMREAD_GRAYSCALE)
    thresh_adp_image = adpThresh(gray_image)
    cv2.imwrite(randomStrThresh + "Thresh.jpg", thresh_adp_image)

    threshImageName = randomStrThresh + "Thresh.jpg"

    ocr = ocrData(threshImageName)

    detection_score_calculation = detectionScoreCalculation(
        detection=detections)

    try:
        os.remove(skewedImageName)
        os.remove(threshImageName)
    except:
        pass
    result = {
        "OCR Data ": ocr,
        "detections": json.loads(str(detection_score_calculation).replace('\'', '\"'))
    }
    return result


if __name__ == "__main__":
    image_path = "Test Cases/aadharVerifyTestCases/1.jpg"
    npimg = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    res = upload_image_aadhar(image)
    print(res)
    print("Done!!!")
