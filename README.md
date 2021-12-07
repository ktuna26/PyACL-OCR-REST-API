# Two-Stage Flask OCR
This is asimple python optical character recognition server. It's model consists of two parts - detection([CRAFT](https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800/acl_craft_pt)) and recognition([deep-text-regonition](https://gitee.com/tianyu__zhou/pyacl_samples/tree/a800/acl_deep_text_recognition_pt))

**Warning!** This is a commercial application by Huawei Turkey (Copyright 2021)

<img alt="teaser" src="./static/img/ocr-postman.png">

<img alt="teaser" src="./static/img/ocr-swagger.png">

## Getting started
Install dependencies;
- numpy
- Pillow
- opencv-python-headless
- scikit-image
- werkzeug
- flask-restplus == 0.12.1
- flask == 1.0.3
- pyarmor
- cryptography

```
pip install -r requirements.txt
```

### Run Server
Open the terminal on the project path and then run the following command.

```bash
python3 app.py
```

**Note :** Import `OCR_REST-API.postman_collection.json` file to `postman` collections for easy demo