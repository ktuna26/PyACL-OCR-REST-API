"""
OCR Rest-API
Copyright 2021 Huawei Technologies Co., Ltd

Usage:
  $ python3 app.py --cfg=data/app.cfg

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
name_space = resutfulApp.namespace('OCR', description='Optical Character Recognition API')


# return the succes message with api
def error_handle(output, status=500, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)

# return the error message with api
def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def threshold_validate(value, field_name):
    min=0.1
    max=1.0
    if not isinstance(value, float):
        raise ValueError("Invalid literal for float(): {0}".format(field_name))
    if min <= value <= max:
        return value
    raise ValueError(f"[{field_name}] must be in range [{min}, {max}]")


# run CRAFT model
def run_craft(image, cropped, text_thresh, link_thresh, low_text):
    print("[INFO] running CRAFT model . . .")

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


#ModelService swagger settings
model_service_param_parser = reqparse.RequestParser()
model_service_param_parser.add_argument('image',  
                         type=werkzeug.datastructures.FileStorage, 
                         location='files', 
                         required=True, 
                         help='Image file')
model_service_param_parser.add_argument('model_name', type=str, help='ocr, craft or text-recog', 
                                        choices=('ocr', 'craft', 'text-recog'), location='path', required=True)
model_service_param_parser.add_argument('link_thresh', type=float, help='Link Confidence Threshold', location='form')
model_service_param_parser.add_argument('low_text', type=float, help='Low-Bound Score', location='form')
model_service_param_parser.add_argument('text_thresh', type=float, help='Text Confidence Threshold', location='form')
model_service_param_parser.add_argument('cropped', type=bool, help='Only valid for test-recog model and it shoul be activated when the image is cropped', location='form')
model_service_param_parser.add_argument('bboxes', type=str, help='Double array which indicates indexes of corners of bounding box. Only required for craft model', location='form')

# run model
@name_space.route('/analyze/<model_name>', methods = ['POST'])
@name_space.expect(model_service_param_parser)
@resutfulApp.doc(responses={
        200: 'Success',
        400: 'Validation Error'
    },description='''
        <h1>OCR: Optical Character Recognition</h1>
        <p>It is the process of converting printed or handwritten texts into digital format. This module is consist of 2 parts which are CRAFT: Character-Region Awareness For Text detection and Deep Text Recognition.</p>
        <br>
        <h2>1) CRAFT: Character-Region Awareness For Text Detection</h2>
        <p>CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.</p>
        <br>
        <img alt="craft" src="./static/img/craft.gif">
        <h2>2) Deep Text Recognition</h2>
        <p>Two-stage Scene Text Recognition (STR), that most existing STR models fit into.</p>
        <br>
        <img alt="deep_text_recog" src="./static/img/deep_text_recog.jpg">
        <h2>Input</h2>
        <p>Supported image types are <b>PNG, JPG and JPEG</b>. Image and model name must be defined. There 3 models which are "craft", "text-recog" and "ocr.". "ocr" model consist of sequenquential call of "craft" and "text-recog" models</p>
        <p>text_threshold: Precision value required for a something to be classified as a letter.</p>
        <p>link_threshold: Amount of distance allowed between two characters for them to be seen as a single word.</p>
        <p>low_text: Amount of boundary space around the letter/word when the coordinates are returned.</p>
        <p>cropped: If only the part containing the text(word) on the image is cropped or if the image consists of only a certain text(word), it should be activated (valid for test-recog model).</p>
        <ul>
        <li>Minimum resoulution must be greater than <b>300x300</b> for cropped parameter is false</li>
        <li>Maximum resoulution must be smaller than <b>300x300</b> for cropped parameter is true</li>
        <li>Model Name must be one of craft, text-recog or ocr</li>
        <li>link_thresh must be in between 0.1~1.0</li>
        <li>low_text must be in between 0.1~1.0</li>
        <li>text_thresh must be in between 0.1~1.0</li>
        <li>cropped can be null or boolean</li>
        </ul>
        <p>"bboxes" parameter is only needed for "text-recog" model. bboxes must be double array which contains bounding box indexes as pixel. Order of the coordinates should be clockwise Sample :        
[
&nbsp;&nbsp;[
&nbsp;&nbsp;&nbsp;&nbsp;0,&nbsp;&nbsp;&nbsp;&nbsp;|x\\top left corner
&nbsp;&nbsp;&nbsp;&nbsp;56,&nbsp;&nbsp;|y/
&nbsp;&nbsp;&nbsp;&nbsp;694,|x\\top right corner
&nbsp;&nbsp;&nbsp;&nbsp;34,&nbsp;&nbsp;|y/
&nbsp;&nbsp;&nbsp;&nbsp;694,|x\\bottom right corner
&nbsp;&nbsp;&nbsp;&nbsp;42,&nbsp;&nbsp;|y/
&nbsp;&nbsp;&nbsp;&nbsp;1,&nbsp;&nbsp;&nbsp;&nbsp;|x\\bottom left corner
&nbsp;&nbsp;&nbsp;&nbsp;64&nbsp;&nbsp;&nbsp;|y/
&nbsp;&nbsp;],
&nbsp;&nbsp;[
&nbsp;&nbsp;&nbsp;&nbsp;99,&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;&nbsp;74,&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;&nbsp;737,|
&nbsp;&nbsp;&nbsp;&nbsp;56,&nbsp;&nbsp;|second
&nbsp;&nbsp;&nbsp;&nbsp;737,|bounding
&nbsp;&nbsp;&nbsp;&nbsp;64,&nbsp;&nbsp;|box
&nbsp;&nbsp;&nbsp;&nbsp;100,|
&nbsp;&nbsp;&nbsp;&nbsp;83&nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;]
        ]
        </p><h2>Returns</h2><p>Accoding to selected model, output will be changed.</p>
        <h3>craft</h3><p>craft returns coordinates of bounding boxes as double array.</p>
        <h3>text-recog and ocr</h3><p>Output of text-recog and ocr models are same. They returns recognized text in below charset: 
        <b>0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ</b><br>
        There is no support for Chinese, Japanese, Arabic (only English).</p>
    '''
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
        cropped = app.cfg.getboolean('model', 'cropped')
        text_thresh = app.cfg.getfloat('model', 'text_thresh')
        link_thresh = app.cfg.getfloat('model', 'link_thresh')
        low_text = app.cfg.getfloat('model', 'low_text')
        
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
        if model_name == 'craft': # text detection
            # run CRAFT model
            boxes_coord = run_craft(img_rgb_plw, cropped, text_thresh, link_thresh, low_text)
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
            # run CRAFT model
            boxes_coord = run_craft(img_rgb_plw, cropped, text_thresh, link_thresh, low_text)
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
                                            "invalid model name, model names must be one of craft, text-recog or ocr"}), status=400)


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
    app.allowed_extensions = app.cfg.get('file', 'allowed_extensions')
    
    # define min. and max. resolution
    app.max_width = app.cfg.getint('file', 'max_width')
    app.max_height = app.cfg.getint('file', 'max_height')
    app.min_width = app.cfg.getint('file', 'min_width')
    app.min_height = app.cfg.getint('file', 'min_height')

    print("[INFO] loading models %s & %s"%(app.cfg.get('model', 'detec_path'), app.cfg.get('model', 'recog_path')))

    # initialize models
    app.detec_model = Model(app.cfg.getint('npu', 'device_id'), path.abspath(app.cfg.get('model', 'detec_path')))
    print("[INFO] type of the detec_model ", str(type(app.detec_model)))
    app.recog_model = Model(app.cfg.getint('npu', 'device_id'), path.abspath(app.cfg.get('model', 'recog_path')), 
                            app.cfg.get('model', 'characters'))
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