from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
import time
from res_manager import ResultManager

def print_cache_train(train_log):
    train_path = train_log["image_path"]
    train_acc_r = train_log["acc_r"]
    train_size_r = train_log["size_r"]
    train_epsilon = train_log["epsilon"]
    print("train_num ",len(train_path))
    print("acc ",np.mean(train_acc_r))
    print("size ",np.mean(train_size_r))
    print("epsilon ",np.mean(train_epsilon))
    print("\n")

def print_cache(face_train):
    image_path = [i[0] for i in list(face_train.items())]
    results = [i[1] for i in list(face_train.items())]
    error_codes = [i["error_code"] for i in list(results)]
    banchmark_qs = [i["banchmark_q"] for i in list(results)]
    print("image_num ",len(image_path))
    print("error_codes ",np.mean(error_codes))
    print("banchmark_qs ",np.mean(banchmark_qs))
    print("error_codes0 ",error_codes.count(0))
    print("error_codes1 ",error_codes.count(1))
    print("error_codes2 ",error_codes.count(2))
    print("banchmark_qs75 ",banchmark_qs.count(75))
    print("\n")

def print_cache_log(inference_log):
    inference_path = inference_log["image_path"]
    inference_status = inference_log["status"]
    inference_accuracy = inference_log["accuracy"]
    inference_size_reward = inference_log["size_reward"]
    inference_action = inference_log["action"]
    print("inference_num ", len(inference_path))
    print("status ", np.mean(inference_status))
    # print("status1 ",inference_status.count(1))
    print("acc ", np.mean(inference_accuracy))
    print("size ", np.mean(inference_size_reward))
    print("action ", np.mean(inference_action))
    print("action100 ", np.mean(inference_action[:100]))
    print("\n")

if __name__=="__main__":
    # ref_cache结果
    with open("evaluation_results/image_reference_cache_face_retrain_DNIM.defaultdict","rb") as file:
        face_retrain_DNIM = pk.load(file)
    # with open("evaluation_results/image_reference_cache_baidu_initial_DNIM.defaultdict","rb") as file:
    #     face_retrain_DNIM = pk.load(file)
    with open("evaluation_results/image_reference_cache_baidu_retrain_DNIM.defaultdict", 'rb') as f:
        ref_cache = pk.load(f)
    with open("evaluation_results/baidu_all_DNIM_imagenet.defaultdict", 'rb') as f:
        ref_cache = pk.load(f)
    print_cache(ref_cache)
    # 提取rm缓存
    # rm = ResultManager('evaluation_results')
    # rm.print_meta_info()
    # 训练的face结果
    # face = rm.load(1)
    # print_cache(face)
    # retain的log结果
    ref = list(ref_cache.items())