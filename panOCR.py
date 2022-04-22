import easyocr

reader = easyocr.Reader(['en','hi'], gpu=True)

def ocrPan(image):
    data = reader.readtext(image, detail=0)

    data_txt = ""

    for ele in data:
        data_txt = data_txt +' '+ ele.lower()
  		
    return data_txt

if __name__ == "__main__":
    image_path = "test_images_pan/1.jpg"
    res = ocrPan(image_path)
    print(res)