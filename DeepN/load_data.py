from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
from deep import DeepN
import time

from src.cloud_apis import AmazonRekognition
from src.cloud_apis import FacePP
from src.cloud_apis import Baidu

if __name__ == '__main__':
    #提取image路径文件
    # with open("image_reference_cache_amazon.defaultdict", "rb") as file:
    #     data = pk.load(file)
    with open("result/imagenet_baidu_ref2000.pkl", "rb") as file:
        data = pk.load(file)
    image_path = []
    data1 = list(data.items())
    image_num = len(data1)
    # image_num = 1
    for i in range(image_num):
        # [0]取出路径和Q值，#分开路径和Q值
        data2 = data1[i][0].split("#", 1)[0]
        # 分出hsli和gnode02
        data3 = data2.split('/', 4)
        # 连接home和文件名
        s = '/'+'/'.join([data3[1], data3[4]])
        image_path.append(s)
    image_num = len(image_path)