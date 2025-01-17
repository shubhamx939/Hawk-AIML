import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import os


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


if __name__ == '__main__':
    image = cv2.imread('test_images_aadhar/6.JPG')
    angle, rotated = correct_skew(image)
    print(angle)
    cv2.imshow('rotated', rotated)
    #cv2.imwrite('rotated.jpg', rotated)
    print("Skewed Image generated")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #os.remove('rotated.jpg')

    
