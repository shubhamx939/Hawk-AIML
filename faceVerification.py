from deepface import DeepFace

def faceVerification(image_main, image_comp):
    models = ["VGG-Face", "Facenet"]
    result_vgg = DeepFace.verify(image_main, image_comp, model_name = models[0])
    result_facenet = DeepFace.verify(image_main, image_comp, model_name = models[1])
    try:
        if result_vgg['verified'] == result_facenet['verified']:
            return result_facenet
        else:
            return result_facenet
    except Exception as e:
        error_json = {
            "Status : " : e
        }
        return error_json

if __name__ == '__main__':
    image1 = "test_images_facial/Anushka Sharma.jpg"
    image2 = "test_images_facial/ANUSHKA_SHARMA.jpg"
    result1, result2 = faceVerification(image_main=image1, image_comp=image2)
    print(result1, result2)
    print("Done!!!") 