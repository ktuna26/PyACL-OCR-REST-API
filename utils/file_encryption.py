"""
Model Encryption Tool
Copyright 2021 Huawei Technologies Co., Ltd

Usage:
  $ python3 file_encryption.py --file <file_name>

CREATED:  2021-12-07 08:12:13
MODIFIED: 2021-12-07 15:48:45
"""
# -*- coding:utf-8 -*-
import argparse

from os import path
from cryptography.fernet import Fernet


# run tool
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='../weights/text_detec.om', help='Dencrypted file path')
    opt = parser.parse_args()
    print("[INFO] starting encryption")

    # key generation
    key = Fernet.generate_key()
    print("[INFO] genereted key : ", key)

    # using the generated key
    fernet = Fernet(key)

    original = None
    # opening the original file to encrypt
    with open(opt.file_path, 'rb') as file:
        original = file.read()

    # encrypting the file
    encrypted = fernet.encrypt(original)

    # opening the file in write mode and
    # writing the encrypted data
    file_name, file_ext = path.splitext(opt.file_path)
    with open(file_name + '_encrypt' + file_ext, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)
    
    print("[INFO] Done!")