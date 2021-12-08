# Docker Start Comand 
# Copyright 2021 Huawei Technologies Co., Ltd
# 
# Usage:
#   $ sudo docker_start.sh <host_port> <container_port> <image_name>
# 
# CREATED:  2021-11-07 15:12:13
# MODIFIED: 2021-12-07 16:48:45

#!/usr/bin/env bash
host_port=$1
container_port=$2
docker_image=$3
docker run -it \
           -p ${host_por}:${container_port} \
           --rm \
           --ipc=host \
           --device=/dev/davinci0 \
           --device=/dev/davinci_manager \
           --device=/dev/devmm_svm \
           --device=/dev/hisi_hdc \
           -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
           -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
           -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
           -v /var/log/npu/slog/:/var/log/npu/slog \
           -v /var/log/npu/profiling/:/var/log/npu/profiling \
           -v /var/log/npu/dump/:/var/log/npu/dump \
           -v /var/log/npu/:/usr/slog \
           -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
           ${docker_image}