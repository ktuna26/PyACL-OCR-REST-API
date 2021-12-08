# Dockerfile
# Copyright 2021 Huawei Technologies Co., Ltd
# 
# Usage:
#   $ sudo docker build -t pyacl_ocr_api:1.0 \
#                       --build-arg NNRT_PKG=Ascend-cann-nnrt_5.0.2_linux-x86_64.run .
# 
# CREATED:  2021-11-24 15:12:13
# MODIFIED: 2021-12-07 16:48:45


#OS and version number. Change them based on the site requirements.
FROM python:3.7.5-slim

# Set the parameters of the offline inference engine package.
ARG NNRT_PKG

# Set environment variables.
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=\
$ASCEND_BASE/driver/lib64:\
$ASCEND_BASE/driver/lib64/common:\
$ASCEND_BASE/driver/lib64/driver:\
$ASCEND_BASE/nnrt/latest/acllib/lib64:\
$LD_LIBRARY_PATH
ENV PYTHONPATH=$ASCEND_BASE/nnrt/latest/pyACL/python/site-packages/acl:\
$PYTHONPATH
ENV ASCEND_OPP_PATH=$ASCEND_BASE/nnrt/latest/opp
ENV ASCEND_AICPU_PATH=\
$ASCEND_BASE/nnrt/latest/x86_64-linux
RUN echo $LD_LIBRARY_PATH && \
    echo $PYTHONPATH && \
    echo $ASCEND_OPP_PATH &&\
    echo $ASCEND_AICPU_PATH

# Copy the offline inference engine package.
COPY $NNRT_PKG .

# Install the offline inference engine package.
RUN umask 0022 && \
    groupadd -g 183426 HwHiAiUser && \
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash && \
    usermod -u 183426 HwHiAiUser && \
    chmod +x ${NNRT_PKG} && \
    ./${NNRT_PKG} --quiet --install && \
    rm ${NNRT_PKG} && \
    . /usr/local/Ascend/nnrt/set_env.sh

# set workdir
WORKDIR /pyacl_ocr_api

# copy user config file into docker image
COPY ./data/app_user.cfg /data/
# copy the project into docker image
COPY ./dist /pyacl_ocr_api

# install the necessary package
RUN cd /pyacl_ocr_api && \
    python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

# set a port
EXPOSE 8500

# run the api
ENTRYPOINT ["python3", "app.py"]