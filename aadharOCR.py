import easyocr

hindi_reader = easyocr.Reader(['en', 'hi'], gpu=True)
assamese_reader = easyocr.Reader(['en', 'as'], gpu=True)
bengali_reader = easyocr.Reader(['en', 'bn'], gpu=True)
kannada_reader = easyocr.Reader(['en', 'kn'], gpu=True)
tamil_reader = easyocr.Reader(['en', 'ta'], gpu=True)
telugu_reader = easyocr.Reader(['en', 'te'], gpu=True)
urdu_reader = easyocr.Reader(['en', 'ur'], gpu=True)

eng_keywords = ['government', 'india','unique', 'identification', 'aadhar', 'authority', 'vid','uidai']
hindi_keywords = ['भारत', 'सरकार', 'प्राधिकरण','पहचान','विशिष्ट','भारतीय', 'आधार','आम', 'आदमी', 'का', 'अधिकार','मेरा', 'मेरी', 'पहचान']
bengali_keywords = ['আম', 'আদমী', 'কা', 'আধিকাৰ', 'মেৰা', 'আধাৰ', 'মেৰী', 'পেহচান', 'আম', 'আদমি']
assamese_keywords = ['কা', 'অধিকার', 'মেরা', 'আধার', 'মেরি', 'পেহচান']
kannada_keywords = ['ಮೇರಾ', 'ಆಧಾರ್', 'ಮೇರಿ', 'ಪೆಹಚಾನ್', 'ಆಮ್', 'ಆದ್ಮಿ', 'ಕಾ', 'ಅಧಿಕಾರ್', 'ಭಾರತ', 'ಸರ್ಕಾರ', 'ನನ್ನ', 'ಆಧಾರ್', 'ನನ್ನ', 'ಗುರುತು']
tamil_keywords = ['மேரா', 'ஆதார்', 'மேரி', 'பெஹ்சான்', 'ஆம்', 'ஆத்மி', 'கா', 'ஆதிகார்', 'எனது', 'ஆதார்', 'எனது', 'அடையாளம்', 'இந்திய', 'அரசாங்கம்']
telugu_keywords = ['ఆమ్', 'ఆద్మీ', 'కా', 'అధికార్', 'మేరా', 'ఆధార్', 'మేరీ', 'పెహచాన్','ఆధార్', 'ఆధార్', 'సామాన్యమానవుడి', 'హక్కు']
urdu_keywords = ['میرا' ,'آدھار' ,'میری' ,'پہچن']

aadhar_keywords = eng_keywords + hindi_keywords + bengali_keywords + assamese_keywords + kannada_keywords + tamil_keywords + telugu_keywords + urdu_keywords

def verifyAadhar(image, counter):
    if counter == 1:
        hindi_data = hindi_reader.readtext(image, detail=0, paragraph = True)
        hindi_data_txt = ""
        for ele in hindi_data:
            hindi_data_txt = hindi_data_txt +' '+ ele.lower()
        return hindi_data_txt

    elif counter == 2:
        kannada_data = kannada_reader.readtext(image, detail=0, paragraph = True)
        kannada_data_txt = ""
        for ele in kannada_data:
            kannada_data_txt = kannada_data_txt +' '+ ele.lower()
        return kannada_data_txt

    elif counter == 3:
        telugu_data = telugu_reader.readtext(image, detail=0, paragraph = True)
        telugu_data_txt = ""
        for ele in telugu_data:
            telugu_data_txt = telugu_data_txt +' '+ ele.lower()
        return telugu_data_txt

    elif counter == 4:
        tamil_data = tamil_reader.readtext(image, detail=0, paragraph = True)
        tamil_data_txt = ""
        for ele in tamil_data:
            tamil_data_txt = tamil_data_txt +' '+ ele.lower()
        return tamil_data_txt

    elif counter == 5:
        bengali_data = bengali_reader.readtext(image, detail=0, paragraph = True)
        bengali_data_txt = ""
        for ele in bengali_data:
            bengali_data_txt = bengali_data_txt +' '+ ele.lower()
        return bengali_data_txt

    elif counter == 6:
        assamese_data = assamese_reader.readtext(image, detail=0, paragraph = True)
        assamese_data_txt = ""
        for ele in assamese_data:
            assamese_data_txt = assamese_data_txt +' '+ ele.lower()
        return assamese_data_txt

    elif counter == 7:
        urdu_data = urdu_reader.readtext(image, detail=0, paragraph = True)
        urdu_data_txt = ""
        for ele in urdu_data:
            urdu_data_txt = urdu_data_txt +' '+ ele.lower()
        return urdu_data_txt

def ocrAadhar(img):
    img_file = img
    count = 0
    flag = 1
    is_aadhar = False
   
    for i in range(7):
        ocr_data = verifyAadhar(image=img_file, counter=flag)
    
        for keyword in aadhar_keywords:
            if keyword in ocr_data:
                count = count + 1

        is_aadhar =  count > 2
    
        if is_aadhar:
            break
        else:
            flag = flag + 1
    
    return is_aadhar
    

if __name__ == "__main__":
    image_path = "test_images_aadhar/6.JPG"
    res = ocrAadhar(image_path)
    print(res)
    print("Done!!!")
