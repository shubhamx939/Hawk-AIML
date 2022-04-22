import cv2
import random
import string
import base64

def randomStringGenerator():
    randomStr = ''.join(random.choices(string.ascii_lowercase, k=10))
    return randomStr

def multiFaceName(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    face_array_base64 = []
    face_name_array = []
    face_region = []
    face_detection_base64 = ""
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        region = [x,y,x+w,y+h]
        face_region.append(region)
        random_faces_str = randomStringGenerator()
        random_faces_str = random_faces_str + "_faces.jpg"
        face_name_array.append(random_faces_str)
        cv2.imwrite(random_faces_str, roi_color)
        with open(random_faces_str, "rb") as image2string:
            converted_string = str(base64.b64encode(image2string.read()))
            face_array_base64.append(converted_string)
            
    random_str = randomStringGenerator()
    random_str = random_str + ".jpg"
    cv2.imwrite(random_str, image)
    with open(random_str, "rb") as image2string:
        face_detection_base64 = str(base64.b64encode(image2string.read()))
         
    
    
    res = {
        "faceName": face_name_array,
        "faceRegion": face_region,
        "face_detection": face_detection_base64,
        "faceArray": face_array_base64
    }
    try:
        return { 'faceName': res["faceName"], "faceRegion":res["faceRegion"], "faceDetection": res["face_detection"], "faceArray": res["faceArray"]}
    except Exception as e:
        pass




if __name__ == "__main__":
    image = "Test Cases/faceCountTestCases/1.jpg"
    res= multiFaceName(imagePath=image)
    print(res)
    print("Done!!!")
