from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
from deep import DeepN
import time

from src.cloud_apis import AmazonRekognition
from src.cloud_apis import FacePP
from src.cloud_apis import Baidu


# with open("result/Origin_Face.defaultdict", 'rb') as file:
#     origin_baidu = pk.load(file)
# with open("result/DeepN_Face.defaultdict", 'rb') as file:
#     deepn_baidu = pk.load(file)
# with open("result/Compare_Face.defaultdict", 'rb') as file:
#     compare_baidu = pk.load(file)

# with open("result/Origin_Baidu.defaultdict", 'rb') as file:
#     origin_baidu = pk.load(file)
# with open("result/DeepN_Baidu.defaultdict", 'rb') as file:
#     deepn_baidu = pk.load(file)
# with open("result/Compare_Baidu.defaultdict", 'rb') as file:
#     compare_baidu = pk.load(file)

with open("result/Origin_Baidu1.defaultdict", 'rb') as file:
    origin_baidu = pk.load(file)
with open("result/DeepN_Baidu1.defaultdict", 'rb') as file:
    deepn_baidu = pk.load(file)
with open("result/Compare_Baidu1.defaultdict", 'rb') as file:
    compare_baidu = pk.load(file)

with open("result/ada_compare_deepn.defaultdict", 'rb') as file:
    ada_compare_deepn = pk.load(file)

origin_sizes = [i[1]['size'] for i in list(origin_baidu.items())]
deepn_sizes = [i[1]['size'] for i in list(deepn_baidu.items())]
compare_result = [i[1]['compare'] for i in list(compare_baidu.items())]
print("error_code number ", compare_result.count(-1))
compare_result = [one for one in compare_result if one != -1]
print("avg origin size ", np.mean(origin_sizes))
print("avg deep size ", np.mean(deepn_sizes))
print("avg accuracy ", np.mean(compare_result))
print(" origin size ", len(origin_sizes))
print(" deep size ", len(deepn_sizes))
print(" accuracy size ", len(compare_result))

# compare_baidu_error = [one for one in list(compare_baidu.items()) if one[1]['compare'] != 1]
