from logging import exception
from deepface import DeepFace

def multiFaceAttribute(image_main):
    try:
        result = DeepFace.analyze(img_path = image_main, actions = ['age', 'gender', 'race', 'emotion'])
        return result
    except:
        result = {
            "message": "Poor quality image for finding face attribute"
            }
        return result
        
if __name__ == "__main__":
    image = "Test Cases/faceAttributeTestCases/1.jpg"
    result = multiFaceAttribute(image_main=image)
    print(result)
    print("Done!!!")