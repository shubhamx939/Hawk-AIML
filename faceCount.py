import stat
from telnetlib import STATUS
import cv2
import random
import string

def randomStringGenerator():
    randomStr = ''.join(random.choices(string.ascii_lowercase, k=10))
    return randomStr

def detector(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    no_of_faces = len(faces)
    
    if no_of_faces > 0:
        status = True
        return no_of_faces, status
     
    else:
        status = False
        return no_of_faces, status


if __name__ == "__main__":
    image = "test_images_facial/2.jpg"
    res, status = detector(image)
    print(res, "      ", status)
    print("Done!!!")
