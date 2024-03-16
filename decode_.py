from flask import Flask, request, jsonify
import os
import cv2
import sys
import re
import csv
import datetime
import math
from dbr import *
#from ..cfg import configure
import numpy as np

app = Flask(__name__)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def decoding_image(absolute_img_path, img_file, reader):
    # Implement the decoding process using Dynamsoft Barcode Reader
    # This function is not provided in the code snippet but you should implement it
    pass

def resizing_images(img,img_file,reader):
    h,w = img.shape[:2]
    ratio = (h+150)/h
    resized= cv2.resize(img,(int(w*ratio),h+150),cv2.INTER_CUBIC)
    key = 'upscaled'
    filename = f"croped-{key}-{img_file}"
    cv2.imwrite("temp/not_even/"+filename,resized)
    cv2.imwrite("temp/not_decoding/"+filename,resized)
    data = decoding_image("temp/not_even/"+filename, filename, reader)
    # print("Decoding in progress...")
    #print(data)
    return data

def adding_padding(img,img_file,reader):
    h,w = img.shape[:2]
    new_h,new_w = 512,512
    if h>=450 or w>=450:
        new_h,new_w = 1024,1024
    white_3d = np.zeros((new_h,new_w,3),np.uint8)
    white_3d.fill(255)
    white_2d = cv2.cvtColor(white_3d,cv2.COLOR_RGB2GRAY)
    center_new_h = new_h//2
    center_new_w = new_w//2

    #print(img_file)
    try:
        white_2d[center_new_h - math.ceil(h/2): center_new_h + math.floor(h/2), center_new_w - math.ceil(w/2) : center_new_w + math.floor(w/2)]=img
    except:
        # print(img_file)
        pass
    #timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    file_name = f"croped-white2d-{img_file}"
    cv2.imwrite("temp/not/"+file_name,white_2d)
    cv2.imwrite("temp/not_decoding/"+file_name,white_2d)
    data = decoding_image("temp/not/"+file_name, file_name, reader)
    #if data == file_name:
    #print(data)
    return data

@app.route('/decode_images', methods=['GET'])
def decode_images():
    # Get folder path from the query parameter
    path_for_folder = request.args.get('folder_path')

    # Initialize the CSV logger
    filename = "temp/CSV_DATA/csv_logger.csv"
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image Filename', 'Decoded String', 'Symbology ID'])
    
        # Decode images in the specified folder
        if os.path.exists(path_for_folder) and os.path.isdir(path_for_folder):
            list_dir = sorted_alphanumeric(os.listdir(path_for_folder))
           # BarcodeReader.init_license(configure.DYLIS)
            reader = BarcodeReader()

            for single_dir in list_dir:
                all_imgs = os.listdir(os.path.join(path_for_folder, single_dir))
                all_imgs = [x for x in all_imgs if x.endswith(('.jpg', '.jpeg', '.png'))]
                all_imgs = [x for x in all_imgs if 'croped' in x]

                for img_file in all_imgs:
                    original_file = img_file[img_file.find("_")+1:len(img_file)-4]
                    absolute_img_path = os.path.join(path_for_folder, single_dir, img_file)
                    img = cv2.imread(absolute_img_path)
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    data_from_dynam = decoding_image(absolute_img_path, img_file, reader)
                    if data_from_dynam == img_file:
                        data_from_dynam = resizing_images(gray_img, img_file, reader)
                        if 'croped-upscaled' in data_from_dynam:
                            data_from_dynam = adding_padding(gray_img, img_file, reader)
                            if 'croped-white2d' in data_from_dynam:
                                data_from_dynam = ''

                    csvwriter.writerow([original_file, data_from_dynam, single_dir])

    return jsonify({'message': 'Decoding process completed'}), 200

if __name__ == '__main__':
    app.run(debug=True)


# import os
# import cv2
# import re
# import csv
# import datetime
# import math
# import sys
# #from ..cfg import configure
# #from pyzbar import pyzbar
# import pandas as pd
# from dbr import *
# import numpy as np
# print("[UPDATE]    decoding started.. ")

# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
#     return sorted(data,key=alphanum_key)

# def resizing_images(img,img_file,reader):
#     h,w = img.shape[:2]
#     ratio = (h+150)/h
#     resized= cv2.resize(img,(int(w*ratio),h+150),cv2.INTER_CUBIC)
#     key = 'upscaled'
#     filename = f"croped-{key}-{img_file}"
#     cv2.imwrite("temp/not_even/"+filename,resized)
#     cv2.imwrite("temp/not_decoding/"+filename,resized)
#     data = decoding_image("temp/not_even/"+filename, filename, reader)
#     # print("Decoding in progress...")
#     #print(data)
#     return data


# def adding_padding(img,img_file,reader):
#     h,w = img.shape[:2]
#     new_h,new_w = 512,512
#     if h>=450 or w>=450:
#         new_h,new_w = 1024,1024
#     white_3d = np.zeros((new_h,new_w,3),np.uint8)
#     white_3d.fill(255)
#     white_2d = cv2.cvtColor(white_3d,cv2.COLOR_RGB2GRAY)
#     center_new_h = new_h//2
#     center_new_w = new_w//2

#     #print(img_file)
#     try:
#         white_2d[center_new_h - math.ceil(h/2): center_new_h + math.floor(h/2), center_new_w - math.ceil(w/2) : center_new_w + math.floor(w/2)]=img
#     except:
#         # print(img_file)
#         pass
#     #timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
#     file_name = f"croped-white2d-{img_file}"
#     cv2.imwrite("temp/not/"+file_name,white_2d)
#     cv2.imwrite("temp/not_decoding/"+file_name,white_2d)
#     data = decoding_image("temp/not/"+file_name, file_name, reader)
#     #if data == file_name:
#     #print(data)
#     return data

# def decoding_image(absolute_img_path,img_file,reader):
#     settings = reader.get_runtime_settings()
#     settings.image_preprocessing_modes[0]= EnumImagePreprocessingMode.IPM_GRAY_EQUALIZE
#     settings.image_preprocessing_modes[1]=EnumImagePreprocessingMode.IPM_GRAY_SMOOTH
#     settings.image_preprocessing_modes[2]=EnumImagePreprocessingMode.IPM_SHARPEN_SMOOTH 
#     settings.image_preprocessing_modes[3]=EnumImagePreprocessingMode.IPM_MORPHOLOGY

#     reader.update_runtime_settings(settings)
#     reader.set_mode_argument("ImagePreprocessingModes", 0, "Sensitivity", "9")
#     reader.set_mode_argument("ImagePreprocessingModes", 1, "SmoothBlockSizeX", "10")
#     reader.set_mode_argument("ImagePreprocessingModes", 1, "SmoothBlockSizeY", "10")
#     reader.set_mode_argument("ImagePreprocessingModes", 2, "SharpenBlockSizeX", "5")
#     reader.set_mode_argument("ImagePreprocessingModes", 2, "SharpenBlockSizeY", "5")
#     reader.set_mode_argument("ImagePreprocessingModes", 3, "MorphOperation", "Close")
#     reader.set_mode_argument("ImagePreprocessingModes", 3, "MorphOperationKernelSizeX", "7")
#     reader.set_mode_argument("ImagePreprocessingModes", 3, "MorphOperationKernelSizeY", "7")
#     results = reader.decode_file(absolute_img_path)
#     if results!=None:
#         for text_result in results:
#             return text_result.barcode_text
#     else:
#         #return ""
#         return img_file

# current_directory = os.getcwd()
# csv_data_file = 'temp/CSV_DATA\csv_logger.csv'
# final_directory = os.path.join(current_directory, r'temp/CSV_DATA')

# os.makedirs(final_directory, exist_ok=True)

# csv_file = os.path.join(current_directory, csv_data_file)

# if(os.path.exists(csv_file) and os.path.isfile(csv_file)):
#   os.remove(csv_file)
#   print(f"[UPDATE]    deleting previous data")
# else:
#   print(f"[UPDATE]    previous data not found")

# path_for_folder = "FOLDER_FOR_CROP"
# # path_for_folder = sys.argv[1]

# if not os.path.exists(path_for_folder) or not os.path.isdir(path_for_folder):
#     print(f"[WARNING]     Invalid folder path provided: {path_for_folder}")
#     sys.exit(1)

# count=0
# folder_for_location = path_for_folder
# # print(f"[UPDATE]    folder path {folder_for_location}")
# list_dir = sorted_alphanumeric(os.listdir(folder_for_location))
# # print(f"[UPDATE]    Directories Uploaded {list_dir}")
# filename = "temp/CSV_DATA\csv_logger.csv"
# # BarcodeReader.init_license("t0074oQAAAGrfUjk7XbBhe2Fwejk4025XMuRYTMZ1jWLhjPRr3eaM+teqhGNQdaV1CdItYLMqE2LnntGB4d+NKn1jTaxmbGTUmDShI7I=")
# #BarcodeReader.init_license(configure.DYLIS)
# reader = BarcodeReader()

# with open(filename, 'a', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['Image Filename', 'Decoded String', 'Symbology ID'])

#     for single_dir in list_dir:
#         all_imgs = os.listdir(os.path.join(folder_for_location,single_dir))
#         all_imgs = [x for x in all_imgs if('.jpg' in x or '.jpeg' in x or '.png' in x)]
#         all_imgs = [x for x in all_imgs if('croped' in x)]

#         for img_file in all_imgs:
#             # print("img_file", img_file)
#             original_file = img_file[img_file.find("_")+1:len(img_file)-4]
#             # print("Original", original_file)
#             absolute_img_path = os.path.join(folder_for_location,single_dir,img_file)
#             img = cv2.imread(absolute_img_path)
#             gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#             data_from_dynam = decoding_image(absolute_img_path,img_file,reader)
#             count+=1
#             if(data_from_dynam==img_file):
#                 data_from_dynam = resizing_images(gray_img,img_file,reader)
#                 if 'croped-upscaled' in data_from_dynam:
#                     data_from_dynam = adding_padding(gray_img, img_file,reader)
#                     if 'croped-white2d' in data_from_dynam:
#                         data_from_dynam =''
#             csvwriter.writerow([original_file,data_from_dynam,single_dir])
#             # csvwriter.writerow(['','','',''])
# # except:
# #     print('we are not able to open file')

# print(f"[UPDATE]    Decode Success")