from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
import time
from res_manager import ResultManager

if __name__ == '__main__':
    with open("inference_log.defaultdict","rb") as file:
        inference_log = pk.load(file)
    inference_path = inference_log["image_path"]
    inference_accuracy = inference_log["accuracy"]
    inference_size_reward = inference_log["size_reward"]
    inference_action = inference_log["action"]
    print("inference_num ",len(inference_path))
    print("acc ",np.mean(inference_accuracy))
    print("size ",np.mean(inference_size_reward))
    print("action ",np.mean(inference_action))
    print("action100 ",np.mean(inference_action[:100]))

    with open("image_reference_cache_face_inference.defaultdict","rb") as file:
        face_inference = pk.load(file)
    print("face_inference_num ",len(face_inference))

    with open("image_reference_cache_face_initial.defaultdict","rb") as file:
        face_initial_DNIM = pk.load(file)
    print("face_initial_DNIM _num ",len(face_initial_DNIM ))
    with open("image_reference_cache_face_retrain_DNIM.defaultdict","rb") as file:
        face_retrain_DNIM = pk.load(file)
    print("face_retrain_DNIM _num ",len(face_retrain_DNIM ))
    with open('baidu_imagenet_model_inference_DNIM.defaultdict', 'rb') as f:
        ref_cache1 = pk.load(f)
    print("ref_num1 ",len(ref_cache1))
    with open('baidu_imagenet_model_inference_DNIM_code.defaultdict', 'rb') as f:
        ref_cache2 = pk.load(f)
    print("ref_num2 ",len(ref_cache2))
    with open('baidu_all_DNIM1.defaultdict', 'rb') as f:
        ref_cache3 = pk.load(f)
    print("ref_num3 ",len(ref_cache3))
    # face_inference_path = face_inference["image_path"]
    # inference_accuracy = face_inference["accuracy"]
    # inference_size_reward = face_inference["size_reward"]
    # inference_action = face_inference["action"]
    # print("face_inference_num ",len(face_inference_path))
    # print("acc ",np.mean(inference_accuracy))
    # print("size ",np.mean(inference_size_reward))
    # print("action ",np.mean(inference_action))
    # print("action100 ",np.mean(inference_action[:100]))

