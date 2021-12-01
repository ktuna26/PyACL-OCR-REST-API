"""
OCR Rest-API
Copyright 2021 Huawei Technologies Co., Ltd

Usage:
  $ python3 app.py --cfg=data/app.cfg

CREATED:  2021-11-24 15:12:13
MODIFIED: 2021-11-27 16:48:45
"""

# -*- coding:utf-8 -*-
import argparse
import numpy as np
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

from PIL import Image
from io import BytesIO
from model.acl import Model
from cv2 import imread, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, imwrite
from os import path, getcwd
from configparser import ConfigParser
from flask import Flask, json, Response, request
from flask_restplus import Api, Resource, reqparse, fields, inputs


# initialize the app
app = Flask(__name__)
resutfulApp = Api(app = app, 
                  version = "1.0", 
                  title = "OCR", 
                  description = "Text detection and recognition from image")
name_space = resutfulApp.namespace('ocr', description='Craft APIs')


# return the succes message with api
def error_handle(output, status=500, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)

# return the error message with api
def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


# run CRAFT model
def run_craft(image, cropped, thresholds):
    print("[INFO] running CRAFT model . . .")

    # convert image format
    img_rgb = np.array(image)

    # run model
    print("[INFO] type of the detec_model" + str(type(app.detec_model)))
    bboxes, polys = app.detec_model.run(img_rgb, cropped = cropped, thresholds = thresholds)

    # get boxes coordinate
    boxes_coord = []
    for poly in polys:
        poly = np.array(poly).astype(np.int32).reshape((-1))
        boxes_coord.append(poly.tolist())

    print("[RESULT] image text boxes coordinate --> ", boxes_coord)
    return boxes_coord

# run Text-Reco model
def run_text_reco(image, cropped, boxes_coord, characters):
    print("[INFO] running Text-Recognition model . . .")

    # convert image format
    img_bgr = cvtColor(np.array(image), COLOR_RGB2BGR)

    # run Text-Reco model
    bboxes = app.recog_model.run(img_bgr, cropped = cropped, boxes_coord = boxes_coord, characters = characters)

    # get text
    texts = ""
    try:
        for b in bboxes:
            texts+=b.get_text() + " "
    except TypeError:
        texts+=""

    print("[RESULT] image texts --> ", texts)
    return texts


# get configiration
@name_space.route('/cfg', methods = ['GET'])
class ConfigurationService(Resource):
    def get(self):
            cfg = {}
            for section in app.cfg.sections():
                cfg.update(dict(app.cfg.items(section)))

            print("[RESULT] api configirations --> ", type(cfg))
            return success_handle(json.dumps({"configirations" : cfg}))


model_service_param_parser = reqparse.RequestParser()
model_service_param_parser.add_argument('image',  
                         type=werkzeug.datastructures.FileStorage, 
                         location='files', 
                         required=True, 
                         help='Image file')
model_service_param_parser.add_argument('model_name', type=str, help='ocr, craft or text-recog', choices=('ocr', 'craft', 'text-recog'), location='path', required=True)
model_service_param_parser.add_argument('link_thresh', type=int, help='Some param', location='form')
model_service_param_parser.add_argument('low_text', type=int, help='Some param', location='form')
model_service_param_parser.add_argument('text_thresh', type=int, help='Some param', location='form')
model_service_param_parser.add_argument('cropped', type=inputs.boolean, default=False, location='form')

# run model
@name_space.route('/analyze/<model_name>', methods = ['POST'])
@name_space.expect(model_service_param_parser)
@resutfulApp.doc(responses={
        200: 'Success',
        400: 'Validation Error'
    },description='''
        <h1>CRAFT: Character-Region Awareness For Text detection & Deep Text Recognition</h1><h2>CRAFT</h2><p>CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.<br>
        <img alt="yeah, just like that" src="./static/craft_example.gif">
        </p><h2>Deep Text Recognition</h2><p>Two-stage Scene Text Recognition (STR), that most existing STR models fit into.<br>
        <img alt="hit me one more time" src="./static/deep_text_reco.jpg"></p><h2>Input</h2><p>Supported image types are <b>PNG, JPG and JPEG</b>. Image and model name must be defined.<ul>
        <li>Minimum resoulution must be greater than <b>300x300</b></li>
        <li>Model Name must be one of craft, text-recog or ocr</li>
        <li>link_thresh must be in between 0.1~1.0</li>
        <li>low_text must be in between 0.1~1.0</li>
        <li>text_thresh must be in between 0.1~1.0</li>
        </ul></p><h2>Returns</h2><p>Recognized text in this charset: 
        <b>0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ</b>
        There is no support for Chinese, Japanese, Arabic (only English).
        </p>
    '''
    )
class ModelService(Resource):
    def post(self, model_name):
        # check image is uploaded with image keyword
        if 'image' not in request.files:
            print("[ERROR] image required")
            return error_handle(json.dumps({"errorMessage" : "image required"}), 400)
        
        print("%s"%(request.files['image']))
        image = request.files['image']

        # chech extension of image
        filename = image.filename
        if '.' in filename and \
           filename.rsplit('.', 1)[1].lower() not in app.file_allowed:
            return error_handle(json.dumps({"errorMessage" : "image format must be one of " + app.file_allowed}), status=400)
        
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
        if width < 300 or height <300 :
            return error_handle(json.dumps({"errorMessage" : "Image resolution must be greater than 300x300"}), status=400)
        
        img_rgb_plw = img.convert('RGB') 

        if model_name == 'craft': # text detection
            if 'cropped' not in request.form:
                print("[ERROR] cropped prameter required")
                return error_handle(json.dumps({"errorMessage" : "cropped parameter required"}), status=400)
            elif 'text_thresh' not in request.form:
                print("[ERROR] text_thresh parameter required")
                return error_handle(json.dumps({"errorMessage" : "text_thresh parameter required"}), status=400)
            elif 'link_thresh' not in request.form:
                print("[ERROR] link_thresh parameter required")
                return error_handle(json.dumps({"errorMessage" : "link_thresh parameter required"}), status=400)
            elif 'low_text' not in request.form:
                print("[ERROR] low_text parameter required")
                return error_handle(json.dumps({"errorMessage" : "low_text parameter required"}), status=400)
            else:
                cropped = request.form.getboolean('cropped')
                thresholds = {"text_thresh" : request.getfloat('text_thresh'), 
                            "link_thresh" : request.getfloat('link_thresh'), 
                            "low_text" : request.getfloat('low_text')}

                print(cropped + "\n" + thresholds)

                # run CRAFT model
                boxes_coord = run_craft(img_rgb_plw, cropped, thresholds)
                return success_handle(json.dumps({"boxesCoordinate" : boxes_coord}))

        elif model_name == 'text-recog': # text recognation
            if 'cropped' not in request.form:
                print("[ERROR] cropped prameter required")
                return error_handle(json.dumps({"errorMessage" : "cropped prameter required"}), status=400)
            elif 'bboxes' not in request.form:
                print("[ERROR] bboxes required")
                return error_handle(json.dumps({"errorMessage" : "bboxes required"}), status=400)
            elif 'characters' not in request.form:
                print("[ERROR] characters required")
                return error_handle(json.dumps({"errorMessage" : "characters required"}), status=400)
            else:
                cropped = request.form.getboolean('cropped')
                bboxes = request.form['bboxes']
                characters = request.form['characters']

                print(cropped + "\n" + bboxes + "\n" + characters)
                
                # read the boxes coordinate in list format
                print("[INFO] loading boxes coordinate . . .")
                boxes_coord = [json.loads('[%s]'%i) for i in request.form['bboxes'].strip('][').split('], [')]

                # run Text-Reco model
                texts = run_text_reco(img_rgb_plw, boxes_coord, characters)
                return success_handle(json.dumps({"imageTexts" : texts}))

        elif model_name == 'ocr': # ocr
            if 'cropped' not in request.form:
                print("[ERROR] cropped prameter required")
                return error_handle(json.dumps({"errorMessage" : "cropped parameter required"}), status=400)
            elif 'text_thresh' not in request.form:
                print("[ERROR] text_thresh parameter required")
                return error_handle(json.dumps({"errorMessage" : "text_thresh parameter required"}), status=400)
            elif 'link_thresh' not in request.form:
                print("[ERROR] link_thresh parameter required")
                return error_handle(json.dumps({"errorMessage" : "link_thresh parameter required"}), status=400)
            elif 'low_text' not in request.form:
                print("[ERROR] low_text parameter required")
                return error_handle(json.dumps({"errorMessage" : "low_text parameter required"}), status=400)
            elif 'characters' not in request.form:
                print("[ERROR] characters required")
                return error_handle(json.dumps({"errorMessage" : "characters required"}), status=400)
            else:
                cropped = request.form.getboolean('cropped')
                characters = request.form['characters']
                thresholds = {"text_thresh" : request.getfloat('text_thresh'), 
                            "link_thresh" : request.getfloat('link_thresh'), 
                            "low_text" : request.getfloat('low_text')}
                
                print(cropped + "\n" + thresholds + "\n" + "\n" + characters)

                # run CRAFT model
                boxes_coord = run_craft(img_rgb_plw, cropped, thresholds)

                if not len(boxes_coord):
                    print("[ERROR] no text detected")
                    return error_handle(json.dumps({"errorMessage" : "no text detected"}), status=500)
                else:
                    # run Text-Reco model
                    texts = run_text_reco(img_rgb_plw, boxes_coord, characters)
                    return success_handle(json.dumps({"imageTexts" : texts}))
        else:
            print("[ERROR] invalid model name")
            return error_handle(json.dumps({"errorMessage" : "invalid model name, model names must be one of craft, text-recog or ocr"}), status=400)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./data/app.cfg', help='Configiration File')
    
    opt = parser.parse_args()
    return opt

def init(opt):
    print("[INFO] loading configurations . . .")
    # define configurations
    app.cfg = ConfigParser()
    app.cfg.read(path.abspath(opt.cfg))
    # define allowed file types
    app.file_allowed = app.cfg.get('file', 'file_allowed')

    print("[INFO] loading models %s & %s"%(app.cfg.get('model', 'detec_path'), app.cfg.get('model', 'recog_path')))

    # initialize models
    app.detec_model = Model(app.cfg.getint('npu', 'device_id'), path.abspath(app.cfg.get('model', 'detec_path')))
    print("[INFO] type of the detec_model ", str(type(app.detec_model)))
    app.recog_model = Model(app.cfg.getint('npu', 'device_id'), path.abspath(app.cfg.get('model', 'recog_path')))
    print("[INFO] type of the recog_model ", str(type(app.recog_model)))


# run api 
if __name__ == "__main__":
    print("[INFO] strating ocr_api . . .")
    
    opt = parse_opt()
    init(opt)
        
    app.run(host = app.cfg.get('server', 'host'), 
            port = app.cfg.get('server', 'port'),
            threaded = False,
            debug = app.cfg.getboolean('server', 'debug'))