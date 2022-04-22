from logging import exception
from deepface import DeepFace

def faceAttribute(image_main):
    try:
        result = DeepFace.analyze(img_path = image_main, actions = ['age', 'gender', 'race', 'emotion'])
        return result
    except Exception as e:
        error_json = {
            "Status : " : e
        }
        return error_json
        
if __name__ == "__main__":
    image = "test_images_facial/Anushka Sharma.jpg"
    result = faceAttribute(image_main=image)
    print(type(result))
    print("Done!!!")