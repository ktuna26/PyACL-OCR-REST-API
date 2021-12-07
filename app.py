"""
OCR Rest-API
Copyright 2021 Huawei Technologies Co., Ltd

Usage:
  $ python3 app.py

CREATED:  2021-11-24 15:12:13
MODIFIED: 2021-12-05 16:48:45
"""
# -*- coding:utf-8 -*-
import argparse
import numpy as np
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

from PIL import Image
from io import BytesIO
from model.acl import Model
from os import path, getcwd, remove
from configparser import ConfigParser
from cryptography.fernet import Fernet
from cv2 import imread, cvtColor, COLOR_RGB2BGR
from flask import Flask, json, Response, request
from flask_restplus import Api, Resource, reqparse, fields, inputs


# initialize flask app
app = Flask(__name__)


# return the succes message with api
def error_handle(output, status=500, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)

# return the error message with api
def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


# threshold validation
def threshold_validate(value, field_name):
    min = app.user_cfg.getint('model', 'min_thresh')
    max = app.user_cfg.getint('model', 'max_thresh')

    if not isinstance(value, float):
        raise ValueError("Invalid literal for float(): {0}".format(field_name))
    if min <= value <= max:
        return value
    raise ValueError(f"[{field_name}] must be in range [{min}, {max}]")


# run Text-Detec model
def run_text_detec(image, cropped, text_thresh, link_thresh, low_text):
    print("[INFO] running text-detec model . . .")

    # convert image format
    img_rgb = np.array(image)

    # run model
    bboxes, polys = app.detec_model.run(img_rgb, cropped = cropped, text_thresh = text_thresh, 
                                        link_thresh = link_thresh, low_text = low_text)

    # get boxes coordinate
    boxes_coord = []
    for poly in polys:
        poly = np.array(poly).astype(np.int32).reshape((-1))
        boxes_coord.append(poly.tolist())

    print("[RESULT] image text boxes coordinate --> ", boxes_coord)
    return boxes_coord

# run Text-Reco model
def run_text_reco(image, cropped, boxes_coord):
    print("[INFO] running Text-Recognition model . . .")

    # convert image format
    img_bgr = cvtColor(np.array(image), COLOR_RGB2BGR)

    # run Text-Reco model
    bboxes = app.recog_model.run(img_bgr, cropped = cropped, boxes_coord = boxes_coord)

    # get text
    texts = ""
    try:
        for b in bboxes:
            texts+=b.get_text() + " "
    except TypeError:
        texts+=""

    print("[RESULT] image texts --> ", texts)
    return texts


# file decryption
def decrypt(file_path, key):
  # using the key
  fernet = Fernet(bytes(key, 'utf-8'))
    
  # opening the encrypted file
  file_name, file_ext = path.splitext(file_path)
  with open(file_name + '_encrypt' + file_ext, 'rb') as enc_file:
      encrypted = enc_file.read()
    
  # decrypting the file
  decrypted = fernet.decrypt(encrypted)
    
  # opening the file in write mode and
  # writing the decrypted data
  with open(file_path, 'wb') as dec_file:
      dec_file.write(decrypted)

# remove decrypted file
def remo_decrypted(file_path):
    if path.exists(file_path):
        remove(file_path)


# define developer configurations
app.dev_cfg = ConfigParser()
dev_cfg = path.abspath('./static/app_dev.cfg')

print("[INFO] loading developer configurations . . .")
decrypt(dev_cfg, 'ONYR0Fq5Y51T7Ua9WEoC2fsY3uAb42YLUL1skjyB-jI=')
app.dev_cfg.read(dev_cfg)
remo_decrypted(dev_cfg)


# initialize flask restful app
resutfulApp = Api(app = app, 
                  version = app.dev_cfg.get('swagger', 'version'), 
                  title = app.dev_cfg.get('swagger', 'title'), 
                  description = app.dev_cfg.get('swagger', 'description1'))
name_space = resutfulApp.namespace('OCR', description = app.dev_cfg.get('swagger', 'description2'))


#ModelService swagger settings
model_service_param_parser = reqparse.RequestParser()
model_service_param_parser.add_argument('image',  
                         type=werkzeug.datastructures.FileStorage, 
                         location='files', 
                         required=True, 
                         help='Image file')
model_service_param_parser.add_argument('model_name', type=str, help='ocr, text-detec or text-recog', 
                                        choices=('ocr', 'text-detec', 'text-recog'), location='path', required=True)
model_service_param_parser.add_argument('link_thresh', type=float, help='Link Confidence Threshold', location='form')
model_service_param_parser.add_argument('low_text', type=float, help='Low-Bound Score', location='form')
model_service_param_parser.add_argument('text_thresh', type=float, help='Text Confidence Threshold', location='form')
model_service_param_parser.add_argument('cropped', type=bool, help='Only valid for test-recog model and it shoul be activated when the image is cropped', location='form')
model_service_param_parser.add_argument('bboxes', type=str, help='Double array which indicates indexes of corners of bounding box. Only required for text-detec model', location='form')

# run model
@name_space.route('/analyze/<model_name>', methods = ['POST'])
@name_space.expect(model_service_param_parser)
@resutfulApp.doc(responses={
        200: 'Success',
        400: 'Validation Error'
    },description = app.dev_cfg.get('swagger', 'description3')
    )
class ModelService(Resource):
    def post(self, model_name):
        # check image is uploaded with image keyword
        if 'image' not in request.files:
            print("[ERROR] image required")
            return error_handle(json.dumps({"errorMessage" : "image required"}), 400)
        
        image = request.files['image']
        print("[INFO] cropped : %s"% image)
        
        # chech extension of image
        filename = image.filename
        if '.' in filename and \
        filename.rsplit('.', 1)[1].lower() not in app.allowed_extensions:
            return error_handle(json.dumps({"errorMessage" : "image format must be one of " + app.allowed_extensions}), status=400)

        # default parameters for model
        cropped = app.user_cfg.getboolean('model', 'cropped')
        text_thresh = app.user_cfg.getfloat('model', 'text_thresh')
        link_thresh = app.user_cfg.getfloat('model', 'link_thresh')
        low_text = app.user_cfg.getfloat('model', 'low_text')
        
        # get optional threshold parameters and cropped parameter from request form
        try:
            if 'cropped' in request.form and model_name == 'text-recog':
                cropped = request.form.get('cropped', type=bool)
            if 'text_thresh' in request.form:
                text_thresh = request.form.get('text_thresh', type=float)
                threshold_validate(text_thresh, 'text_thresh')
            if 'link_thresh' in request.form:
                link_thresh = request.form.get('link_thresh', type=float)
                threshold_validate(link_thresh, 'link_thresh')
            if 'low_text' in request.form:
                low_text = request.form.get('low_text', type=float)
                threshold_validate(low_text, 'low_text')
        except ValueError as err:
            return error_handle(json.dumps({"errorMessage" : "{0}".format(err)}), status=400)
        print("[INFO] cropped : %s, text_thresh : %s, link_thresh : %s, low_text%s"%(cropped, text_thresh, 
                                                                                    link_thresh, low_text))
        
        # read the image in PIL format
        print("[INFO] loading image . . .")
        img = image.read()
        # convert image format
        try:
            img = Image.open(BytesIO(img))
        except:
            return error_handle(json.dumps({"errorMessage" : "Uploaded file is not a valid image"}), status=400)
        
        # check min resolution
        width, height = img.size
        if cropped == True:
            if width > app.max_width or height > app.max_height :
                return error_handle(json.dumps({"errorMessage" : "Cropped image resolution must be smaller than 300x300"}), status=400)
        else:
            if width < app.min_width or height < app.min_height :
                return error_handle(json.dumps({"errorMessage" : "Image resolution must be greater than 300x300"}), status=400)
        
        img_rgb_plw = img.convert('RGB')
        if model_name == 'text-detec': # text detection
            # run Text-Detec model
            boxes_coord = run_text_detec(img_rgb_plw, cropped, text_thresh, link_thresh, low_text)
            return success_handle(json.dumps(boxes_coord))
        elif model_name == 'text-recog': # text recognation
            boxes_coord = None
            print("[INFO] cropped : %s"% cropped)

            if not cropped and 'bboxes' not in request.form:
                print("[ERROR] bboxes or cropped (if image cropped) required")
                return error_handle(json.dumps({"errorMessage" : "bboxes or cropped (if image cropped) required"}), status=400)
            elif cropped and 'bboxes' in request.form:
                print("[ERROR] bboxes and cropped can't be enable at the same time")
                return error_handle(json.dumps({"errorMessage" : "bboxes and cropped can't be enable at the same time"}), status=400)
            elif not cropped and 'bboxes' in request.form:
                bboxes = request.form['bboxes']
                print("[INFO] bboxes : %s"% bboxes)
                print("[INFO] loading boxes coordinate . . .")
                boxes_coord = json.loads(bboxes)

            # run Text-Reco model
            texts = run_text_reco(img_rgb_plw, cropped, boxes_coord)
            return success_handle(json.dumps({"imageTexts" : texts}))   
        elif model_name == 'ocr': # ocr
            # run Text-Detec model
            boxes_coord = run_text_detec(img_rgb_plw, cropped, text_thresh, link_thresh, low_text)
            if not len(boxes_coord):
                print("[ERROR] no text detected")
                return error_handle(json.dumps({"errorMessage" : "no text detected"}), status=500)
            else:
                # run Text-Reco model
                texts = run_text_reco(img_rgb_plw, cropped, boxes_coord)
                return success_handle(json.dumps({"imageTexts" : texts}))
        else:
            print("[ERROR] invalid model name")
            return error_handle(json.dumps({"errorMessage" : \
                                            "invalid model name, model names must be one of text-detec, text-recog or ocr"}), status=400)


def init():
    # define user configurations
    print("[INFO] loading user configurations . . .")
    app.user_cfg = ConfigParser()
    app.user_cfg.read(app.dev_cfg.get('file', 'user_cfg'))
    
    # define allowed file types
    app.allowed_extensions = app.user_cfg.get('file', 'allowed_extensions')
    
    # define min. and max. resolution
    app.max_width = app.user_cfg.getint('file', 'max_width')
    app.max_height = app.user_cfg.getint('file', 'max_height')
    app.min_width = app.user_cfg.getint('file', 'min_width')
    app.min_height = app.user_cfg.getint('file', 'min_height')

    # initialize models
    device_id = app.user_cfg.getint('npu', 'device_id')
    print("[INFO] using NPU-%d . . ."% device_id)
    detec_model = path.abspath(app.dev_cfg.get('file', 'detec_path'))
    recog_model = path.abspath(app.dev_cfg.get('file', 'recog_path'))
    print("[INFO] loading models %s & %s"%(detec_model, recog_model))
    
    decrypt(detec_model, '_6ed4PCyeZrguAmMM8vzHme4t0IziAHED6UDTW0tcLY=')
    app.detec_model = Model(device_id, detec_model)
    remo_decrypted(detec_model)
    print("[INFO] type of the detec_model ", str(type(app.detec_model)))

    decrypt(recog_model, 'u8V42WQ6FaJfYy2S1UIZh41B-iu1EX7ams3NzX1J-Jk=')
    app.recog_model = Model(device_id, recog_model, app.user_cfg.get('model', 'characters'))
    remo_decrypted(recog_model)
    print("[INFO] type of the recog_model ", str(type(app.recog_model)))


# run api 
if __name__ == "__main__":
    print("[INFO] strating ocr_api . . .")
    init()
        
    app.run(host = app.user_cfg.get('server', 'host'), 
            port = app.user_cfg.get('server', 'port'),
            threaded = False,
            debug = app.user_cfg.getboolean('server', 'debug'))