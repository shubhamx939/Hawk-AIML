from deepface import DeepFace
import os

def faceRecognition(image_main):
    try: 
        models = ["VGG-Face", "Facenet"]
        image_db = "imageDB/"
        df_vgg = DeepFace.find(img_path = image_main, db_path = image_db, model_name = models[0])
        df_facenet = DeepFace.find(img_path = image_main, db_path = image_db, model_name = models[1])
        if df_vgg['identity'][0] == df_facenet['identity'][0]:
            return df_vgg['identity'][0]
        else:
            res = "Unknown Face Found"
            return res
    except:
        error_json = {
            "Status" : "Not a Face photo!"
        }
        return error_json

if __name__ == "__main__":
    image_path = "Test Cases/faceRecognitionTestCases/1.jpg"
    result = faceRecognition(image_main=image_path)
    print(result)
    result = os.path.basename(result)
    result_name = os.path.splitext(result)[0]
    print(result_name)
    
    print("Done!!!")