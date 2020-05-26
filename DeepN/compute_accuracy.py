from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
from deep import DeepN
import time

from src.cloud_apis import AmazonRekognition
from src.cloud_apis import FacePP
from src.cloud_apis import Baidu

with open("result/Origin_Baidu1.defaultdict", 'rb') as file:
    origin_baidu = pk.load(file)
with open("result/DeepN_Baidu1.defaultdict", 'rb') as file:
    deepn_baidu = pk.load(file)
# with open("result/Compare_Baidu1.defaultdict", 'rb') as file:
#     compare_baidu = pk.load(file)

# origin_baidu = list(origin_baidu.items())
# deepn_baidu = list(deepn_baidu.items())
#
# for i in range(len(deepn_baidu)):
#     deepn_baidu[i]
compare_cache = defaultdict(dict)
image_path = [i[0] for i in list(origin_baidu.items())]
results1 = [i[1]['results'] for i in list(origin_baidu.items())]
deepn_results1 = [i[1]['results'] for i in list(deepn_baidu.items())]
error_code1 = [i[1]['error_code'] for i in list(origin_baidu.items())]
deepn_error_code1 = [i[1]['error_code'] for i in list(deepn_baidu.items())]

# cache = self.cache["%s##%s" % (img_path, quality)]
# error_code = cache['error_code']
# reg_results = cache['results']
# size = cache['size']
for i in range(len(deepn_baidu)):
    results = results1[i]
    deepn_results = deepn_results1[i]
    error_code = error_code1[i]
    deepn_error_code = deepn_error_code1[i]
    if error_code == 0:
        if deepn_error_code == 0:
            # gt_id = np.argmax([line['score'] for line in deepn_results])
            gt_id = np.argmax([line['score'] for line in results])
            gt_label = results[gt_id]['keyword']
            gt_confidence = results[gt_id]['score']
            if not gt_label in [line['keyword'] for line in deepn_results]:
                # deepn返回的标签集都没有真实标签，记为0
                compare_cache["%s" % image_path[i]] = {"compare": 0}
            else:
                compare_cache["%s" % image_path[i]] = {"compare": 1}
        else:
            # deepn识别不出记为0
            compare_cache["%s" % image_path[i]] = {"compare": 0}
    else:
        # origin识别不出记为-1
        compare_cache["%s" % image_path[i]] = {"compare": -1}
    with open("result/Compare_Baidu1.defaultdict", "wb") as file:
        pk.dump(compare_cache, file)

with open("result/Compare_Baidu1.defaultdict", 'rb') as file:
    compare_baidu = pk.load(file)
compare_result = [i[1]['compare'] for i in list(compare_baidu.items())]
compare_result = [one for one in compare_result if one != -1]
print("avg accuracy ", np.mean(compare_result))
print(" accuracy size ", len(compare_result))

