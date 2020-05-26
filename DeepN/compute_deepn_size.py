from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
from deep import DeepN
import time
from numpy import *
import numpy as np
import os
from io import BytesIO
from keras.preprocessing import image
import cv2
import hashlib
import time
import matplotlib.pyplot as plt
import base64
import pickle as pk

from src.cloud_apis import AmazonRekognition
from src.cloud_apis import FacePP
from src.cloud_apis import Baidu

def get_image_path(data):
    image_path = []
    data1 = list(data.items())
    for i in range(len(data1)):
        # [0]取出路径和Q值，#分开路径和Q值
        data2 = data1[i][0].split("#", 1)[0]
        # 分出hsli和gnode02
        data3 = data2.split('/', 4)
        # 连接home和文件名
        s = '/'+'/'.join([data3[1], data3[4]])
        image_path.append(s)
    image_num = len(image_path)
    print("图片数量 ",image_num)
    return image_path,image_num

def zig_zag_flatten(a):
    return np.concatenate([np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-a.shape[0], a.shape[0])])

def file_size(image, quality):
    f = BytesIO()
    image.save(f, format='JPEG', quality = quality)
    file_size = len(f.getvalue())
    return file_size

def compute_size(origin_image,q_tables):
    pil_qtables = {}
    for idx, q_table in enumerate(q_tables):
        pil_qtables[idx] = zig_zag_flatten(q_table).tolist()
    f = BytesIO()
    origin_image.save(f, format='JPEG', qtables=pil_qtables)
    deepn_file_size = len(f.getvalue())
    origin_image_size = file_size(origin_image,75)
    f.seek(0)
    deepn_image = Image.open(f)
    return origin_image_size,deepn_file_size

if __name__ == '__main__':
    with open("result/imagenet_baidu_ref2000.pkl", "rb") as file:
        data = pk.load(file)
    image_path,image_num = get_image_path(data)
    # 导入Q表
    q_tables1 = np.load("result/q_tables.npy")
    origins = []
    deepns = []
    for i in image_path:
        origin_image = Image.open(i)
        origin_image_size,deepn_file_size = compute_size(origin_image,q_tables1)
        origins.append(origin_image_size)
        deepns.append(deepn_file_size)
    print("avg origin",np.mean(origins))
    print("avg deepn",np.mean(deepns))