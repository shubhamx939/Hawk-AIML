from email import message
from logging import debug, error, exception
from xxlimited import foo
from flask import Flask, jsonify, request
import re
import json
import numpy
import cv2
import PIL
from PIL import Image
import os
import string
import random
import base64
import time
from datetime import datetime
from numpyencoder import NumpyEncoder
import pandas as pd

from faceVerification import faceVerification
from faceRecognition import faceRecognition
from faceAttribute import faceAttribute
from faceCount import detector
from ocr import ocrData
from panVerify import upload_image_pan
from aadharVerify import upload_image_aadhar
from panDocumentIdentification import upload_image_pan_doc
from aadharDocumentIdentification import upload_image_aadhar_doc
from getAllFaces import faceDetector
from multiFaceName import multiFaceName
from multiFaceAttribute import multiFaceAttribute
from antispoofing import antiSpoofing
from faceFound import faceFounded

app = Flask(__name__)

code_execution_path = "codeExecution/"
imageDB = "imageDB/"

os.chdir(code_execution_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'JPG'}

def allowedFile(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def randomStringGenerator():
    randomStr = ''.join(random.choices(string.ascii_lowercase, k=10))
    return randomStr


def randomStringGeneratorWithTimestamp():
    random_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    ts = str(round(time.time() * 1000))
    randomStr = ts + "_" + random_str
    return randomStr


def base64ToImage(base64_img, random_string):
    base64_img_bytes = base64_img.encode('utf-8')
    with open(random_string, 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)




####################################### SINGLE FACE ATTRIBUTE ROUTE ###########################################
@app.route('/single-face-attribute', methods=['POST'])
def singleFacialAttribute():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files):
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name + '.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result = faceAttribute(image_main=random_string_name)
                    res = {
                        "code": "200",
                        "message": "Image detected successfully!",
                        "result": {
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "No face found in the particular image!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0:
            base64string = request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string)
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name + '.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result = faceAttribute(image_main=random_string_name)
                    res = {
                        "code": "200",
                        "message": "Image detected successfully!",
                        "result": {
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "No face found in the particular image!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END SINGLE FACE ATTRIBUTE ROUTE #######################################

####################################### FACE RECOGNITION ROUTE ################################################
@app.route('/face-recognition', methods=['POST'])
def facialRecognition():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files) :
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                data = Image.fromarray(image)
                data.save('face_reco.png')
                try:
                    result = faceRecognition(image_main="face_reco.png")
                    result = os.path.basename(result)
                    result_name = os.path.splitext(result)[0]
                    res = {
                        "code": "200",
                        "message": "Face Recognition done successfully!",
                        "result":{
                            "name":result_name
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "Face is not matching with anyone from Image Database!",
                        "result":{
                            "name": {
                                "message": "Face is not matching with anyone from Image Database!"
                            }
                        }
                    }
                    return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 :
            base64string =  request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string) 
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                try:
                    result = faceRecognition(image_main=image)
                    result = os.path.basename(result)
                    result_name = os.path.splitext(result)[0]
                    res = {
                        "code": "200",
                        "message": "Face Recognition done successfully!",
                        "result":{
                            "name":result_name
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "Face is not matching with anyone from Image Database!",
                        "result":{
                            "name": {
                                "message": "Face is not matching with anyone from Image Database!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END FACE RECOGNITION ROUTE ############################################


####################################### FACE VERIFICATION ROUTE ###############################################
@app.route('/face-verification', methods=['POST'])
def facialVerification():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file1' in request.files):
            if ('file2' in request.files):
                file1 = request.files['file1']
                file2 = request.files['file2']
                if (file1 and allowedFile(file1.filename) and file2 and allowedFile(file2.filename)):
                    npimg1 = numpy.fromfile(file1, numpy.uint8)
                    npimg2 = numpy.fromfile(file2, numpy.uint8)
                    image1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
                    image2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)
                    random_string_name1 = randomStringGeneratorWithTimestamp()
                    random_string_name2 = randomStringGeneratorWithTimestamp()
                    random_string_name1= random_string_name1+'.jpeg'
                    random_string_name2= random_string_name2+'.jpeg'
                    cv2.imwrite(random_string_name1, image1)
                    cv2.imwrite(random_string_name2, image2)
                    try:
                        result_facenet = faceVerification(
                            image_main=random_string_name1, image_comp=random_string_name2)
                        res = {
                            "code": "200",
                            "message": "Face Verification done successfully!",
                            "result": {
                                "data": result_facenet
                            }
                        }
                        return jsonify(res), 200
                    except:
                        res = {
                            "code": "200",
                            "message": "No face found in the particular image!",
                            "result": {
                                "data": {
                                    "message": "No face found in the particular image!"
                                }
                            }
                        }                     
                        return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 and 'base64_image2' in request_data and request_data['base64_image2'] != 0:
            base64string1 = request_data['base64_image1']
            base64string2 = request_data['base64_image2']
            if len(base64string1) != 0 and len(base64string2) != 0:
                random_string1 = randomStringGenerator()
                random_string2 = randomStringGenerator()
                random_string1 = random_string1 + ".png"
                random_string2 = random_string2 + ".png"
                base64ToImage(base64string1, random_string1)
                base64ToImage(base64string2, random_string2)
                base64_img1 = random_string1
                base64_img2 = random_string2
                npimg1 = numpy.fromfile(base64_img1, numpy.uint8)
                npimg2 = numpy.fromfile(base64_img2, numpy.uint8)
                image1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
                image2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)
                random_string_name1 = randomStringGeneratorWithTimestamp()
                random_string_name2 = randomStringGeneratorWithTimestamp()
                random_string_name1= random_string_name1+'.jpeg'
                random_string_name2= random_string_name2+'.jpeg'
                cv2.imwrite(random_string_name1, image1)
                cv2.imwrite(random_string_name2, image2)
                try:
                    result_facenet = faceVerification(
                        image_main=random_string_name1, image_comp=random_string_name2)
                    res = {
                        "code": "200",
                        "message": "Face Verification done successfully!",
                        "result": {
                            "data": result_facenet
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "No face found in the particular image!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }                     
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END FACE VERIFICATION ROUTE ###########################################

####################################### FACE COUNT ROUTE ######################################################
@app.route('/face-count', methods=['POST'])
def facialCount():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files) :
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    result, status = detector(image_path)
                    res = {
                        "code": "200",
                        "message": "Face Count done successfully!",
                        "result": {
                            "faceFound": status,
                            "numberOfFaces": result
                        }
                    }
                    return jsonify(res), 200
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "No",
                        "result": {
                            "faceFound": {
                                "message": "No face found in the particular image!"
                            },
                            "numberOfFaces": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 :
            base64string =  request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string) 
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    result, status = detector(image_path)
                    res = {
                        "code": "200",
                        "message": "Face Count done successfully!",
                        "result": {
                            "faceFound": status,
                            "numberOfFaces": result
                        }
                    }
                    return jsonify(res), 200
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "No",
                        "result": {
                            "faceFound": {
                                "message": "No face found in the particular image!"
                            },
                            "numberOfFaces": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END FACE COUNT ROUTE ##################################################

####################################### GET ALL FACES ROUTE ###################################################
@app.route('/get-all-faces', methods=['POST'])
def getFaces():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files) :
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    result = faceDetector(image_path)
                    res = {
                        "code": "200",
                        "message": "All faces extraction done successfully!",
                        "result": {
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "Face extraction can't be done!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 :
            base64string =  request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string) 
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    result = faceDetector(image_path)
                    res = {
                        "code": "200",
                        "message": "All faces extraction done successfully!",
                        "result": {
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "Face extraction can't be done!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END GET ALL FACES ROUTE ###############################################

####################################### FACE ATTRIBUTES ROUTE #################################################
@app.route('/face-attribute', methods=['POST'])
def faceAttributes():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files) :
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    result = multiFaceName(image_path)
                    faces = result["faceName"]
                    attribute_arr = []
                    face_count = 0
                    for face in faces:
                        face_count = face_count + 1
                        try:
                            attributes = multiFaceAttribute(face)
                            attribute_arr.append(attributes)  
                        except exception as e:
                            pass            
                    result_df = pd.Series(result["faceRegion"]).to_json(orient='values')
                    res = {
                        "code": "200",
                        "message": "Image detected successfully!",
                        "result": {
                            "region": result_df,
                            "data": attribute_arr,
                            "totalFaces": face_count,
                            "totalFaceAttributes": len(attribute_arr),
                            "faceDetection": result["faceDetection"],
                            "faceArray": result["faceArray"]
                        }
                    }
                    return jsonify(res), 200
                
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "Image detected successfully!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 :
            base64string =  request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string) 
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    result = multiFaceName(image_path)
                    faces = result["faceName"]
                    attribute_arr = []
                    face_count = 0
                    for face in faces:
                        face_count = face_count + 1
                        try:
                            attributes = multiFaceAttribute(face)
                            attribute_arr.append(attributes)  
                        except exception as e:
                            pass            
                    result_df = pd.Series(result["faceRegion"]).to_json(orient='values')      
                    res = {
                        "code": "200",
                        "message": "Image detected successfully!",
                        "result": {
                            "region": result_df,
                            "data": attribute_arr,
                            "totalFaces": face_count,
                            "totalFaceAttributes": len(attribute_arr),
                            "faceDetection": result["faceDetection"],
                            "faceArray": result["faceArray"]
                        }
                    }
                    return jsonify(res), 200
                
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "Image detected successfully!",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END FACE ATTRIBUTES ROUTE #############################################

####################################### FACE ANTI SPOOFING ROUTE ##############################################
@app.route('/anti-spoofing', methods=['POST'])
def facialAntiSpoofing():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files) :
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    face_found = faceFounded(image_path)
                    
                    if face_found == 1:
                        result = antiSpoofing(image_path)
                        res = {
                            "code": "200",
                            "message": "Face Anti-Spoofing done successfully!",
                            "result": {
                                "data": result,
                            }
                        }
                        return jsonify(res), 200
                    elif face_found == 0:
                        res = {
                            "code": "200",
                            "message": "No",
                            "result": {
                                "data": {
                                    "message": "No face found in the particular image!"
                                },
                            }
                        }
                        return jsonify(res), 200  
                    elif face_found > 1:
                        res = {
                            "code": "200",
                            "message": "No",
                            "result": {
                                "data": {
                                    "message": "Multiple face found in the particular image!"
                                },
                            }
                        }
                        return jsonify(res), 200                        
                        
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "No",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            },
                        }
                    }
                    return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 :
            base64string =  request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string) 
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGenerator()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                image_path = os.path.abspath(random_string_name)
                try:
                    face_found = faceFounded(image_path)
                    
                    if face_found == 1:
                        result = antiSpoofing(image_path)
                        res = {
                            "code": "200",
                            "message": "Face Anti-Spoofing done successfully!",
                            "result": {
                                "data": result,
                            }
                        }
                        return jsonify(res), 200
                    elif face_found == 0:
                        res = {
                            "code": "200",
                            "message": "No",
                            "result": {
                                "data": {
                                    "message": "No face found in the particular image!"
                                },
                            }
                        }
                        return jsonify(res), 200  
                    elif face_found > 1:
                        res = {
                            "code": "200",
                            "message": "No",
                            "result": {
                                "data": {
                                    "message": "Multiple face found in the particular image!"
                                },
                            }
                        }
                        return jsonify(res), 200                     
                except Exception as e:
                    res = {
                        "code": "200",
                        "message": "No",
                        "result": {
                            "data": {
                                "message": "No face found in the particular image!"
                            },
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END FACE ANTI SPOOFING ROUTE ##########################################

####################################### OCR ROUTE #############################################################
@app.route('/ocr-image', methods=['POST'])
def ocrImage():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files) :
            file = request.files['file']
            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                try: 
                    ocr_data = ocrData(image)
                    ocr_result = str(ocr_data)
                    res ={
                        "code": "200",
                        "message": "OCR done successfully!",
                        "result": {
                            "data": ocr_result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "OCR can't be done on this image!",
                        "result":{
                            "data":{
                                "message": "OCR can't be done on this image!"
                            }
                        }
                    }
                    return jsonify(res), 200


        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0 :
            base64string =  request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string) 
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                try: 
                    ocr_data = ocrData(image)
                    ocr_result = str(ocr_data)
                    res ={
                        "code": "200",
                        "message": "OCR done successfully!",
                        "result": {
                            "data": ocr_result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "OCR can't be done on this image!",
                        "result":{
                            "data":{
                                "message": "OCR can't be done on this image!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END OCR ROUTE #########################################################


####################################### PAN VERIFY ROUTE ######################################################
@app.route('/pan-verify', methods=['POST'])
def panVerification():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files):
            file = request.files['file']

            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result = upload_image_pan(image)
                    res = {
                        "code": "200",
                        "message": "Pan Card verification done successfully!",
                        "result":{
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "Pan Card verification can't be done!",
                        "result":{
                            "data": {
                                "message": "Pan Card verification can't be done!"
                            }
                        }
                    }
                    return jsonify(res), 200



        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0:
            base64string = request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string)
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result = upload_image_pan(image)
                    res = {
                        "code": "200",
                        "message": "Pan Card verification done successfully!",
                        "result":{
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "Pan Card verification can't be done!",
                        "result":{
                            "data": {
                                "message": "Pan Card verification can't be done!"
                            }
                        }
                    }
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END PAN VERIFY ROUTE ##################################################

####################################### AADHAAR VERIFY ROUTE ##################################################
@app.route('/aadhaar-verify', methods=['POST'])
def aadharVerification():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files):
            file = request.files['file']

            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result = upload_image_aadhar(image)
                    res = {
                        "code": "200",
                        "message": "Aadhar Card verification done successfully!",
                        "result":{
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "aadhar Card verification can't be done!",
                        "result":{
                            "data": {
                                "message": "aadhar Card verification can't be done!"
                            }
                        }
                    }            
                    return jsonify(res), 200


        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0:
            base64string = request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string)
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result = upload_image_aadhar(image)
                    res = {
                        "code": "200",
                        "message": "Aadhar Card verification done successfully!",
                        "result":{
                            "data": result
                        }
                    }
                    return jsonify(res), 200
                except:
                    res = {
                        "code": "200",
                        "message": "aadhar Card verification can't be done!",
                        "result":{
                            "data": {
                                "message": "aadhar Card verification can't be done!"
                            }
                        }
                    }              
                    return jsonify(res), 200
        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END AADHAAR VERIFY ROUTE ##############################################

####################################### DOCUMENT VERIFICATION ROUTE ###########################################
@app.route('/document-verify', methods=['POST'])
def documentVerification():
    if request.method == 'POST':
        request_data = request.get_json()
        if ('file' in request.files):
            file = request.files['file']

            if (file and allowedFile(file.filename)):
                npimg = numpy.fromfile(file, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result_pan = upload_image_pan_doc(image)
                    result_aadhar = upload_image_aadhar_doc(
                        image)
                    if result_pan['Pan Card Found'] == True:
                        res = {
                            "code": "200",
                            "message": "Pan Card verification done successfully!",
                            "result": {
                                "data": result_pan
                            }
                        }
                        return jsonify(res), 200
                    elif result_aadhar['Aadhar Card Found'] == True:
                        res = {
                            "code": "200",
                            "message": "Aadhar Card verification done successfully!",
                            "result": {
                                "data": result_aadhar
                            }
                        }
                        return jsonify(res), 200
                    else:
                        res = {
                            "code": "200",
                            "message": "Unknown Document found!",
                            "result": {
                                "data": {
                                    "message": "Unknown Document found!"
                                }
                            }
                        }
                        return jsonify(res), 200
                except:
                        res = {
                            "code": "200",
                            "message": "Document verification can't be done!",
                            "result": {
                                "data": {
                                    "message": "Document verification can't be done!"
                                }
                            }
                        }
                        return jsonify(res), 200

        elif request_data is not None and 'base64_image1' in request_data and request_data['base64_image1'] != 0:
            base64string = request_data['base64_image1']
            if len(base64string) != 0:
                random_string = randomStringGenerator()
                random_string = random_string + ".png"
                base64ToImage(base64string, random_string)
                base64_img = random_string
                npimg = numpy.fromfile(base64_img, numpy.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                random_string_name = randomStringGeneratorWithTimestamp()
                random_string_name = random_string_name+'.jpeg'
                cv2.imwrite(random_string_name, image)
                try:
                    result_pan = upload_image_pan_doc(image)
                    result_aadhar = upload_image_aadhar_doc(
                        image)
                    if result_pan['Pan Card Found'] == True:
                        res = {
                            "code": "200",
                            "message": "Pan Card verification done successfully!",
                            "result": {
                                "data": result_pan
                            }
                        }
                        return jsonify(res), 200
                    elif result_aadhar['Aadhar Card Found'] == True:
                        res = {
                            "code": "200",
                            "message": "Aadhar Card verification done successfully!",
                            "result": {
                                "data": result_aadhar
                            }
                        }
                        return jsonify(res), 200
                    else:
                        res = {
                            "code": "200",
                            "message": "Unknown Document found!",
                            "result": {
                                "data": {
                                    "message": "Unknown Document found!"
                                }
                            }
                        }
                        return jsonify(res), 200
                except:
                        res = {
                            "code": "200",
                            "message": "Document verification can't be done!",
                            "result": {
                                "data": {
                                    "message": "Document verification can't be done!"
                                }
                            }
                        }
                        return jsonify(res), 200

        else:
            res = {
                "code": "400",
                "message": "Bad Request!"
            }
            return jsonify(res), 400
    res = {
        "code": "400",
        "message": "Request method is not allowed!"
    }
    return jsonify(res), 400

####################################### END DOCUMENT VERIFICATION ROUTE #######################################


if __name__ == "__main__":
    app. run(host="192.168.100.134",port=5000)


"""app. run(host="192.168.100.134", port=5000)"""
