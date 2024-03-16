import os
import cv2
import datetime
import csv
import re
import torch
import math
import sys
import time
import numpy as np
#from ..cfg import configure

sys.path.append('yolov7')

from pathlib import Path
import torch.backends.cudnn as cudnn
from numpy import random
from model.expriment import attempt_load
from yolov7 import LoadStreams, LoadImages
from yolov7 import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7 import select_device, load_classifier, time_synchronized, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def model_loading(img_path):
    opt  = {
    
    "weights": "yolov7/runs/train/best_v1.pt", # Path to weights file default weights are for nano model
   # "yaml"   : "Trash-5/data.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.5, # confidence threshold for inference.
    "iou-thres" : 0.50, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : None  # list of classes to filter or None

    }


    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        img0 = cv2.imread(img_path)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

    # Inference
        t1 = time_synchronized()
        pred = model(img, augment= False)[0]

    # Apply NMS
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:

                classes.append(opt['classes'].index(class_name))

        coordinates = []
        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # print(det[:,:4])
                # print(len(det))
                #print(det)
                for count in range(len(det)):
                    xmin = int(det[count][0])
                    ymin = int(det[count][1])
                    xmax = int(det[count][2])
                    ymax = int(det[count][3])
                    coordinates.append((xmin,ymin,xmax,ymax))
    return coordinates       


def crop_random_part(key,folder_path,w,x,y,z,crop_folder) -> None:
    # key is image file name
    #folder_path is directory name in which file is present
    images = os.path.join(folder_path,key)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if images.endswith(('.tiff', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG')):        
        image = cv2.imread(images,2)
    try:
        cropped_image = image[x-15:z+15,w-15:y+15]
        # file_name = f"croped-{key}_{serial_number}.jpg"
        file_name = f"croped-{timestamp}_{key}.jpg"
        save_path = os.path.join(crop_folder, file_name)
        cv2.imwrite(save_path,cropped_image)
    except:
        cropped_image = image[x:z,w:y]
        # file_name = f"croped_{key}_{serial_number}.jpg"
        file_name = f"croped-{timestamp}_{key}.jpg"
        save_path = os.path.join(crop_folder, file_name)
        cv2.imwrite(save_path,cropped_image)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key=alphanum_key)

# folder_for_not_crop = "test_image" # folder to upload images
folder_for_not_crop = "FOLDER_FOR_NOT_CROP"
folder_for_crop = "FOLDER_FOR_CROP"
folder_for_deteced = "FOLDER_FOR_DETECED"
list_dir = sorted_alphanumeric(os.listdir(folder_for_not_crop))
for single_dir in list_dir:
    imgs_name = sorted_alphanumeric(os.listdir(os.path.join(folder_for_not_crop,single_dir)))
    imgs_name = [x for x in imgs_name if('crop' not in x )]

    for single_img in imgs_name:
        dest = os.path.join(folder_for_not_crop,single_dir)
        dete = os.path.join(folder_for_deteced,single_dir)
        crop_folder = os.path.join(folder_for_crop,single_dir)

        img0 = cv2.imread(os.path.join(dest,single_img))
        coordinates =model_loading(os.path.join(dest,single_img))  #folder_for_not_crop+str(single_dir)+'/'+single_img)
        # print(coordinates)
        # total_objects = str(len(coordinates))
        # print("Total Objects",total_objects )
        for _ in range(len(coordinates)):
           # print(_)
            xmin = coordinates[_][0]
            ymin = coordinates[_][1]
            xmax = coordinates[_][2]
            ymax = coordinates[_][3]
            crop_random_part(single_img, dest, xmin, ymin, xmax, ymax,crop_folder)
        #drawing bound box
        for _ in range(len(coordinates)):
           # print(_)
            xmin = coordinates[_][0]
            ymin = coordinates[_][1]
            xmax = coordinates[_][2]
            ymax = coordinates[_][3]
            detected_image = cv2.rectangle(img0, (xmin,ymin), (xmax,ymax), (0,255,0), 10)
            cv2.imwrite(os.path.join(dete,'detected_'+single_img),detected_image)
        print(single_img+" total objects:  "+str(len(coordinates)))
        # total object = 
        if len(coordinates) == 0:
            cv2.imwrite(os.path.join(crop_folder,f'croped_{single_img}.jpg'),img0)
            print(f"Qr and Barcode not detected in {single_img}")
    
        # for j in range(len(coordinates)):
        #     # xmin = math.floor((points_dataframe.iloc[j][0]))
        #     # ymin = math.floor((points_dataframe.iloc[j][1]))
        #     # xmax = math.ceil(points_dataframe.iloc[j][2])
        #     # ymax = math.ceil(points_dataframe.iloc[j][3])
        #     print(coordinates[j][0],coordinates[j][1],coordinates[j][2],coordinates[j][3])
        #     crop_random_part(single_img, dest, coordinates[j][0],coordinates[j][1],coordinates[j][2],coordinates[j][3])

#print(list_dir)
print("[UPDATE]    IMAGE PROCESSING COMPLETED.. ")



















# from flask import Flask, request, jsonify
# import os
# import cv2
# import datetime
# import csv
# import re
# import torch
# import math
# import sys
# import time
# import numpy as np
# import random
# #from ..cfg import configure

# sys.path.append('yolov7')
# app = Flask(__name__)
# from pathlib import Path
# import torch.backends.cudnn as cudnn
# from numpy import random

# # Define the folder paths
# folder_for_not_crop = "test_image"  # Update with your folder path
# folder_for_crop = "cropped_images"  # Update with your folder path
# folder_for_detected = "detected_images"  # Update with your folder path

# @app.route('/process_images', methods=['POST'])
# def process_images():
#     if 'images' not in request.files:
#         return jsonify({'error': 'No images uploaded'}), 400

#     # Save the uploaded images
#     uploaded_images = request.files.getlist('images')
#     for img in uploaded_images:
#         img.save(os.path.join(folder_for_not_crop, img.filename))

#     # Process the uploaded images
#     process_images_in_folder(folder_for_not_crop)

#     return jsonify({'message': 'Image processing completed'}), 200


# def model_loading(img_path):
#     opt = {

#         "weights": "yolov7/runs/train/best_v1.pt",  # Path to weights file default weights are for nano model
#         # "yaml"   : "Trash-5/data.yaml",
#         "img-size": 640,  # default image size
#         "conf-thres": 0.5,  # confidence threshold for inference.
#         "iou-thres": 0.50,  # NMS IoU threshold for inference.
#         "device": 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
#         "classes": None  # list of classes to filter or None

#     }

#     with torch.no_grad():
#         weights, imgsz = opt['weights'], opt['img-size']
#         set_logging()
#         device = select_device(opt['device'])
#         half = device.type != 'cpu'
#         model = attempt_load(weights, map_location=device)  # load FP32 model
#         stride = int(model.stride.max())  # model stride
#         imgsz = check_img_size(imgsz, s=stride)  # check img_size
#         if half:
#             model.half()

#         names = model.module.names if hasattr(model, 'module') else model.names
#         colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#         if device.type != 'cpu':
#             model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

#         img0 = cv2.imread(img_path)
#         img = letterbox(img0, imgsz, stride=stride)[0]
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Inference
#         t1 = time_synchronized()
#         pred = model(img, augment=False)[0]

#         # Apply NMS
#         classes = None
#         if opt['classes']:
#             classes = []
#             for class_name in opt['classes']:
#                 classes.append(opt['classes'].index(class_name))

#         coordinates = []
#         pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
#         t2 = time_synchronized()
#         for i, det in enumerate(pred):
#             s = ''
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
#             if len(det):
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
#                 # print(det[:,:4])
#                 # print(len(det))
#                 # print(det)
#                 for count in range(len(det)):
#                     xmin = int(det[count][0])
#                     ymin = int(det[count][1])
#                     xmax = int(det[count][2])
#                     ymax = int(det[count][3])
#                     coordinates.append((xmin, ymin, xmax, ymax))
#     return coordinates


# def process_images_in_folder(folder_path):
#     list_dir = sorted_alphanumeric(os.listdir(folder_path))
#     for single_dir in list_dir:
#         imgs_name = sorted_alphanumeric(os.listdir(os.path.join(folder_path, single_dir)))
#         imgs_name = [x for x in imgs_name if ('crop' not in x)]

#         for single_img in imgs_name:
#             dest = os.path.join(folder_path, single_dir)
#             detected = os.path.join(folder_for_detected, single_dir)
#             crop_folder = os.path.join(folder_for_crop, single_dir)

#             img0 = cv2.imread(os.path.join(dest, single_img))
#             coordinates = model_loading(os.path.join(dest, single_img))

#             for _ in range(len(coordinates)):
#                 xmin = coordinates[_][0]
#                 ymin = coordinates[_][1]
#                 xmax = coordinates[_][2]
#                 ymax = coordinates[_][3]
#                 crop_random_part(single_img, dest, xmin, ymin, xmax, ymax, crop_folder)

#             for _ in range(len(coordinates)):
#                 xmin = coordinates[_][0]
#                 ymin = coordinates[_][1]
#                 xmax = coordinates[_][2]
#                 ymax = coordinates[_][3]
#                 detected_image = cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
#                 cv2.imwrite(os.path.join(detected, 'detected_' + single_img), detected_image)

#             print(single_img + " total objects:  " + str(len(coordinates)))

#             if len(coordinates) == 0:
#                 cv2.imwrite(os.path.join(crop_folder, f'croped_{single_img}.jpg'), img0)
#                 print(f"Qr and Barcode not detected in {single_img}")

#     print("[UPDATE]    IMAGE PROCESSING COMPLETED.. ")
# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)




# def crop_random_part(key, folder_path, w, x, y, z, crop_folder) -> None:
#     # key is image file name
#     # folder_path is directory name in which file is present
#     images = os.path.join(folder_path, key)
#     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
#     if images.endswith(('.tiff', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG')):
#         image = cv2.imread(images, 2)
#     try:
#         cropped_image = image[x - 15:z + 15, w - 15:y + 15]
#         # file_name = f"croped-{key}_{serial_number}.jpg"
#         file_name = f"croped-{timestamp}_{key}.jpg"
#         save_path = os.path.join(crop_folder, file_name)
#         cv2.imwrite(save_path, cropped_image)
#     except:
#         cropped_image = image[x:z, w:y]
#         # file_name = f"croped_{key}_{serial_number}.jpg"
#         file_name = f"croped-{timestamp}_{key}.jpg"
#         save_path = os.path.join(crop_folder, file_name)
#         cv2.imwrite(save_path, cropped_image)


# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(data, key=alphanum_key)


# # folder_for_not_crop = "test_image" # folder to upload images

# list_dir = sorted_alphanumeric(os.listdir(folder_for_not_crop))
# for single_dir in list_dir:
#     imgs_name = sorted_alphanumeric(os.listdir(os.path.join(folder_for_not_crop, single_dir)))
#     imgs_name = [x for x in imgs_name if ('crop' not in x)]

#     for single_img in imgs_name:
#         dest = os.path.join(folder_for_not_crop, single_dir)
#         dete = os.path.join(folder_for_detected, single_dir)
#         crop_folder = os.path.join(folder_for_crop, single_dir)

#         img0 = cv2.imread(os.path.join(dest, single_img))
#         coordinates = model_loading(
#             os.path.join(dest, single_img))  # folder_for_not_crop+str(single_dir)+'/'+single_img)
#         # print(coordinates)
#         # total_objects = str(len(coordinates))
#         # print("Total Objects",total_objects )
#         for _ in range(len(coordinates)):
#             # print(_)
#             xmin = coordinates[_][0]
#             ymin = coordinates[_][1]
#             xmax = coordinates[_][2]
#             ymax = coordinates[_][3]
#             crop_random_part(single_img, dest, xmin, ymin, xmax, ymax, crop_folder)
#         # drawing bound box
#         for _ in range(len(coordinates)):
#             # print(_)
#             xmin = coordinates[_][0]
#             ymin = coordinates[_][1]
#             xmax = coordinates[_][2]
#             ymax = coordinates[_][3]
#             detected_image = cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
#             cv2.imwrite(os.path.join(dete, 'detected_' + single_img), detected_image)
#         print(single_img + " total objects:  " + str(len(coordinates)))
#         # total object =
#         if len(coordinates) == 0:
#             cv2.imwrite(os.path.join(crop_folder, f'croped_{single_img}.jpg'), img0)
#             print(f"Qr and Barcode not detected in {single_img}")

#         # for j in range(len(coordinates)):
#         #     # xmin = math.floor((points_dataframe.iloc[j][0]))
#         #     # ymin = math.floor((points_dataframe.iloc[j][1]))
#         #     # xmax = math.ceil(points_dataframe.iloc[j][2])
#         #     # ymax = math.ceil(points_dataframe.iloc[j][3])
#         #     print(coordinates[j][0],coordinates[j][1],coordinates[j][2],coordinates[j][3])
#         #     crop_random_part(single_img, dest, coordinates[j][0],coordinates[j][1],coordinates[j][2],coordinates[j][3])

# # print(list_dir)
# print("[UPDATE]    IMAGE PROCESSING COMPLETED.. ")

# if __name__ == '__main__':
#     app.run(debug=True)  # Run the Flask app in debug mode


