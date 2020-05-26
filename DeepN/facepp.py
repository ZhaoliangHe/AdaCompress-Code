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

def compare(deep_results, results, ref_size):
    gt_id = np.argmax([line['score'] for line in deep_results])
    gt_label = deep_results[gt_id]['keyword']
    gt_confidence = deep_results[gt_id]['score']
    if not gt_label in [line['keyword'] for line in results]:
        return 0, 0, ref_size
    else:
        return 0, 1, ref_size

def zig_zag_flatten(a):
    return np.concatenate([np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-a.shape[0], a.shape[0])])

def file_size(image, quality):
    f = BytesIO()
    image.save(f, format='JPEG', quality = quality)
    file_size = len(f.getvalue())
    return file_size

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

if __name__ == '__main__':
    start = time.time()
    #提取image路径文件
    # with open("image_reference_cache_amazon.defaultdict", "rb") as file:
    #     data = pk.load(file)
    with open("result/imagenet_baidu_ref2000.pkl", "rb") as file:
        data = pk.load(file)
    image_path,image_num = get_image_path(data)
    # 导入Q表
    q_tables1 = np.load("result/q_tables.npy")
    print(q_tables1)
    pil_qtables = {}
    for idx, q_table in enumerate(q_tables1):
        pil_qtables[idx] = zig_zag_flatten(q_table).tolist()
    #上传原图到云服务器
    cloud_agent = FacePP()
    # cloud_agent = Baidu()
    # cloud_agent = AmazonRekognition()
    ref_quality = 75
    cache = defaultdict(dict)
    deepn_cache = defaultdict(dict)
    compare_cache = defaultdict(dict)
    # 导入之前结果
    # with open("result/Origin_Baidu.defaultdict", 'rb') as file:
    #     cache = pk.load(file)
    # with open("result/DeepN_Baidu.defaultdict", 'rb') as file:
    #     deepn_cache = pk.load(file)
    # with open("result/Compare_Baidu.defaultdict", 'rb') as file:
    #     compare_cache = pk.load(file)
    num1 = image_num
    # num1 = 10
    for i in range(num1):
        image = Image.open(image_path[i])
        origin_image = image
        # resize
        # image_size = min(image.size)
        # crop_image = image.crop((0, 0, image_size, image_size))
        # deepn
        f = BytesIO()
        origin_image.save(f, format='JPEG', qtables=pil_qtables)
        # deepn_file_size = len(f.getvalue())
        # print("origin_image_size", file_size(origin_image, 75))
        # print("deepn_file_size", deepn_file_size)
        f.seek(0)
        deepn_image = Image.open(f)
        # deepn_image = DeepN(crop_image)
        # sever
        error_code, results, size = cloud_agent.recognize(origin_image, ref_quality)
        deepn_error_code, deepn_results, deepn_size = cloud_agent.recognize(deepn_image, ref_quality)
        #保存结果
        cache["%s" % image_path[i]] = {"error_code": error_code,
                                       "results": results,
                                       "size": size}
        # with open("result/Origin_Baidu1.defaultdict", "wb") as file:
        with open("result/Origin_Face.defaultdict", "wb") as file:
        # with open("result/Origin_Aws.defaultdict", "wb") as file:
             pk.dump(cache, file)
        #保存结果
        deepn_cache["%s" % image_path[i]] = {"error_code": deepn_error_code,
                                       "results": deepn_results,
                                       "size": deepn_size}
        # with open("result/DeepN_Baidu1.defaultdict", "wb") as file:
        with open("result/DeepN_Face.defaultdict", "wb") as file:
        # with open("result/DeepN_Aws.defaultdict", "wb") as file:
             pk.dump(deepn_cache, file)
        #compare
        if error_code == 0:
            if deepn_error_code == 0:
                # gt_id = np.argmax([line['score'] for line in deepn_results])
                # gt_label = deepn_results[gt_id]['keyword']
                # gt_confidence = deepn_results[gt_id]['score']
                gt_id = np.argmax([line['score'] for line in results])
                gt_label = results[gt_id]['keyword']
                gt_confidence = results[gt_id]['score']
                # if not gt_label in [line['keyword'] for line in results]:
                if not gt_label in [line['keyword'] for line in deepn_results]:
                    # deepn返回的标签集都没有真实标签，记为0
                    compare_cache["%s" % image_path[i]] = {"compare": 0, "ref_results":results, "deepn_results":deepn_results,
                                                           "ref_size": size, "deepn_size": deepn_size}
                else:
                    compare_cache["%s" % image_path[i]] = {"compare": 1, "ref_results":results, "deepn_results":deepn_results,
                                                           "ref_size": size, "deepn_size": deepn_size}
            else:
                # deepn识别不出记为0
                compare_cache["%s" % image_path[i]] = {"compare": 0, "ref_results": results,
                                                       "deepn_results": deepn_results,
                                                       "ref_size": size, "deepn_size": deepn_size}
        else:
            # origin识别不出记为-1
            compare_cache["%s" % image_path[i]] = {"compare": -1, "ref_results": results, "deepn_results": deepn_results,
                                                   "ref_size": size, "deepn_size": deepn_size}
        # with open("result/Compare_Baidu1.defaultdict", "wb") as file:
        with open("result/Compare_Face.defaultdict", "wb") as file:
        # with open("result/Compare_Aws.defaultdict", "wb") as file:
             pk.dump(compare_cache, file)

    # with open("result/Origin_Baidu1.defaultdict", 'rb') as file:
    #     origin_baidu = pk.load(file)
    # with open("result/DeepN_Baidu1.defaultdict", 'rb') as file:
    #     deepn_baidu = pk.load(file)
    # with open("result/Compare_Baidu1.defaultdict", 'rb') as file:
    #     compare_baidu = pk.load(file)
    with open("result/Origin_Face.defaultdict", 'rb') as file:
        origin_baidu = pk.load(file)
    with open("result/DeepN_Face.defaultdict", 'rb') as file:
        deepn_baidu = pk.load(file)
    with open("result/Compare_Face.defaultdict", 'rb') as file:
        compare_baidu = pk.load(file)
    # with open("result/Origin_Aws.defaultdict", 'rb') as file:
    #     origin_baidu = pk.load(file)
    # with open("result/DeepN_Aws.defaultdict", 'rb') as file:
    #     deepn_baidu = pk.load(file)
    # with open("result/Compare_Aws.defaultdict", 'rb') as file:
    #     compare_baidu = pk.load(file)

    # 输出结果
    origin_sizes = [i[1]['size'] for i in list(origin_baidu.items())]
    deepn_sizes = [i[1]['size'] for i in list(deepn_baidu.items())]
    compare_result = [i[1]['compare'] for i in list(compare_baidu.items())]
    print("error_code number ", compare_result.count(-1))
    compare_result = [one for one in compare_result if one != -1]
    print("avg origin size ",np.mean(origin_sizes))
    print("avg deep size ",np.mean(deepn_sizes))
    print("avg accuracy ",np.mean(compare_result))
    print("origin size ", len(origin_sizes))
    print("deep size ", len(deepn_sizes))
    print("accuracy size ", len(compare_result))

    end = time.time()
    print("runing time", end - start)
