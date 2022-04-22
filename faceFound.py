import cv2

def faceFounded(imagePath):
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
            
    return no_of_faces


if __name__ == "__main__":
    image = "Test Cases/faceCountTestCases/1.jpg"
    res = faceFounded(imagePath=image)
    print(res)
    print("Done!!!")
