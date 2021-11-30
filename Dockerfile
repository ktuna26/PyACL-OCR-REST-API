FROM ascendhub.huawei.com/public-ascendhub/infer-modelzoo:21.0.2

COPY ./requirements.txt /pyacl_ocr_api/requirements.txt
WORKDIR /pyacl_ocr_api
RUN pip install -r requirements.txt
COPY . /pyacl_ocr_api

ENV LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/sdk_home/mxManufacture/lib:/usr/local/sdk_home/mxManufacture/opensource/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/nnrt/latest/acllib/lib64:/usr/local/Ascend/atc/lib64:/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH"

RUN echo $LD_LIBRARY_PATH

EXPOSE 8500
CMD ["python3", "app.py"]
