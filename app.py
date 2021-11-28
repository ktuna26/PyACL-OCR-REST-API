"""
OCR Rest-API
Copyright 2021 Huawei Technologies Co., Ltd

Usage:
  $ python3 app.py --host=0.0.0.0 \
                --port=9687 \
                --detec-model=weights/craft.om \
                --recog-model=weights/None-ResNet-None-CTC.om --device-id=0 \
                --cfg=data/app.cfg

CREATED:  2021-11-24 15:12:13
MODIFIED: 2021-11-27 16:48:45
"""

# -*- coding:utf-8 -*-
import argparse
import numpy as np

from PIL import Image
from io import BytesIO
from model.acl import Model
from cv2 import imread, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, imwrite
from os import path, getcwd
from configparser import ConfigParser
from flask import Flask, json, Response, request


# initialize the app
thresholds = {"text_thresh":0.7, "link_thresh":0.4, "low_text":0.4}
model1 = Model(0, './weights/craft.om', thresholds = thresholds)
model2 = Model(0, './weights/None-ResNet-None-CTC.om', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
app = Flask(__name__)


# return the succes message with api
def error_handle(output, code=1, status=500, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)

# return the error message with api
def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


# run CRAFT model
def run_craft(image):
    print("[INFO] running CRAFT model . . .")

    # read the image in PIL format
    print("[INFO] loading image . . .")
    img = image.read()
    # convert image format
    img_rgb_plw = Image.open(BytesIO(img)).convert('RGB') 
    img_rgb = np.array(img_rgb_plw)

    # run model
    bboxes, polys = model1.run(img_rgb)

    # get boxes coordinate
    boxes_coord = []
    for poly in polys:
        poly = np.array(poly).astype(np.int32).reshape((-1))
        boxes_coord.append(poly.tolist())

    print("[RESULT] image text boxes coordinate --> ", boxes_coord)
    return boxes_coord

# run Text-Reco model
def run_text_reco(image, boxes_coord):
    print("[INFO] running Text-Recognition model . . .")

    # read the image in PIL format
    print("[INFO] loading image . . .")
    img = image.read()
    # convert image format
    img_rgb_plw = Image.open(BytesIO(img))
    img_bgr = cvtColor(np.array(img_rgb_plw), COLOR_RGB2BGR)

    # run Text-Reco model
    bboxes = model2.run(img_bgr, boxes_coord, cropped = app.cfg.getboolean('model', 'cropped'))

    # get text
    texts = ""
    for b in bboxes:
        texts+=b.get_text() + " "

    print("[RESULT] image texts --> ", texts)
    return texts


# OCR homepage
@app.route('/ocr', methods = ['GET'])
def ocr_home_page():
    print("This is a simple OCR API\n[Copyright 2021 Huawei Technologies Co., Ltd]")
    return success_handle(json.dumps({"Mesagge" : "Hello! Welcome to Huawei OCR API."}))

# CRAFT homepage
@app.route('/CRAFT', methods = ['GET'])
def craft_home_page():
    print("CRAFT : Character-Region Awareness For Text Detection & Deep Text Detection\n \
    [https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800/acl_craft_pt]")
    return success_handle(json.dumps({"Mesagge" : "Hello! Welcome to Huawei Text Detector API."}))

# Text-Recog homepage
@app.route('/text-recog', methods = ['GET'])
def recog_home_page():
    print("PyTorch Deep Text Recognition\n \
    [https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800/acl_deep_text_recognition_pt]")
    return success_handle(json.dumps({"Mesagge" : "Hello! Welcome to Huawei Text Recognizer API."}))
    

# get configiration
@app.route('/get-cfg', methods = ['GET'])
def get_cfg():
    if request.method == 'GET':
        cfg = app.cfg._sections['model']

        print("[RESULT] api configirations --> ", cfg)
        return success_handle(json.dumps({"status" : True, "configirations" : cfg}))

# set configiration
@app.route('/set-cfg', methods = ['POST'])
def set_cfg():
    if request.method == 'POST':
        if 'chracters' not in request.form and \
            'cropped' not in request.form and \
            'text_thresh' not in request.form and \
            'link_thresh' not in request.form and \
            'low_text' not in request.form:
            print("[ERROR] at least one of the 'chracters', 'cropped', 'text_thresh', 'link_thresh' and 'low_text' elements required")
            return error_handle(json.dumps({"status" : False, "failMesagge" : "at least one of the 'chracters', \
                                'cropped', 'text-thresh', 'link-thresh' and 'low-tex' elements required."}))
        else :
            print(request.form)

            for name in request.form:
                # get name from the form 
                app.cfg['model'][name] = request.form.get(name) # to do -> add a patern for charecter seting !
            
            print("[INFO] configirations has been saved in the cfg file")
            return success_handle(json.dumps({"status" : True}))


# text detection
@app.route('/CRAFT/analyze', methods = ['POST'])
def detec():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("[ERROR] image required")
            return error_handle(json.dumps({"status" : False, "failMesagge" : "image required"}))
        else:
            print("%s"%(request.files['image']))
            image = request.files['image']

            # check allowed file extension
            if image.mimetype not in app.file_allowed:
                print("[ERROR] file extension is not allowed")
                return error_handle(json.dumps({"status" : False, "failMesagge" : "only files ends with *.png, *.jpg, *.jpeg can be upload."}))
            else:
                # run CRAFT model
                boxes_coord = run_craft(image)

                return success_handle(json.dumps({"status" : True, "boxesCoordinate" : boxes_coord}))

# text recognation
@app.route('/text-recog/analyze', methods = ['POST']) # to do add boxes_coord for post
def recog():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("[ERROR] image required")
            return error_handle(json.dumps({"status" : False, "failMesagge" : "image required"}))
        else:
            print("%s"%(request.files['image']))
            image = request.files['image']

            # check allowed file extension
            if image.mimetype not in app.file_allowed:
                print("[ERROR] file extension is not allowed . . .")
                return error_handle(json.dumps({"status" : False, "failMesagge" : "only files ends with *.png, *.jpg, *.jpeg can be upload."}))
            else:
                if 'bboxes' not in request.form:
                    print("[ERROR] bboxes required")
                    return error_handle(json.dumps({"status" : False, "failMesagge" : "bboxes required"}))
                else:
                    print("%s"%(request.form['bboxes']))
                    
                    # read the boxes coordinate in list format
                    print("[INFO] loading boxes coordinate . . .")
                    boxes_coord = [json.loads('[%s]'%i) for i in request.form['bboxes'].strip('][').split('], [')]

                    # run Text-Reco model
                    texts = run_text_reco(image, boxes_coord)

                    return success_handle(json.dumps({"status" : True, "imageTexts" : texts}))

# ocr
@app.route('/ocr/analyze', methods = ['POST'])
def ocr():
    if request.method == 'POST':
        if 'image' not in request.files:
            print("[ERROR] image required")
            return error_handle(json.dumps({"status" : False, "failMesagge" : "image required"}))
        else:
            print("%s"%(request.files['image']))
            image = request.files['image']

            # check allowed file extension
            if image.mimetype not in app.file_allowed:
                print("[ERROR] file extension is not allowed")
                return error_handle(json.dumps({"status" : False, "failMesagge" : "only files ends with *.png, *.jpg, *.jpeg can be upload."}))
            else:
                # run CRAFT model
                boxes_coord = run_craft(image)

                # run Text-Reco model
                texts = run_text_reco(image, boxes_coord)
                
                return success_handle(json.dumps({"status" : True, "imageTexts" : texts}))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',  type=str, default='0.0.0.0', help='API Host Address')
    parser.add_argument('--port', type=str, default='8500', help='API Port Number')
    parser.add_argument('--cfg', type=str, default='./data/app.cfg', help='Configiration File')
    parser.add_argument('--detec-model', type=str, default='./weights/craft.om', help='CRAFT Acl Model')
    parser.add_argument('--recog-model', type=str, default='./weights/None-ResNet-None-CTC.om', help='Text-Recognition Acl Model')
    parser.add_argument('--device-id', type=int, default=0, help='Huawei NPU Device Id')
    
    opt = parser.parse_args()
    return opt

def init(opt):
    # define configirations
    app.cfg = ConfigParser()
    app.cfg.read(path.abspath(opt.cfg))

    # define allowed file types
    app.file_allowed = (app.cfg.get('file', 'file_allowed')).split(', ')
    # creat thresholds dictionary
    thresholds = {"text_thresh" : app.cfg.getfloat('model', 'text_thresh'), 
                "link_thresh" : app.cfg.getfloat('model', 'link_thresh'), 
                "low_text" : app.cfg.getfloat('model', 'low_text')}

    # initialize models
    # model = Model(opt.device_id, opt.detec_model, thresholds = thresholds)
    # model2 = Model(opt.device_id, opt.recog_model, app.cfg.get('model', 'characters'))


# run api 
if __name__ == "__main__":
    print("[INFO] strating ocr_api . . .")
    
    opt = parse_opt()
    init(opt)
        
    app.run(host = opt.host, 
            port = opt.port, 
            debug = app.cfg.getboolean('server', 'debug'), 
            threaded = app.cfg.getboolean('server', 'threaded'))