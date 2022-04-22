import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image
import os


def adpThresh(img_read):
    retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)

    retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)

    img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    return img_thresh_adp



if __name__ == '__main__':
    image_path = "Test Cases/faceAttributeTestCases/2.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_thresh_adp = adpThresh(image)
    cv2.imshow("Img_Thresh_ADP", img_thresh_adp)

    #cv2.imwrite("Thresh_ADP_Img.jpg", img_thresh_adp)
    print("Adaptive Threshold Image generated")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
