from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
import time
# from res_manager import ResultManager
import os
import psutil

process = psutil.Process(os.getpid())
print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')

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

def compute_att(result,att_name):
    results = [i[1] for i in list(result.items())]
    att = [i[att_name] for i in list(results)]
    att_mean = np.mean(att)
    print("the number of %s is %d " % (att_name,len(att)))
    return att_mean
# att_name = "error_code" "ref_size" "size" "banchmark_q"

if __name__=="__main__":
    # with open('compute_time_results/baidu_ref_results.defaultdict', 'rb') as f:
    #     ref_results = pk.load(f)
    # with open('compute_time_results/baidu_compress_results.defaultdict', 'rb') as f:
    #     compress_results = pk.load(f)
    # with open('compute_time_results/baidu_train_log.defaultdict', 'rb') as f:
    #     train_log = pk.load(f)
    # with open('compute_time_results/baidu_compress_results_inference.defaultdict', 'rb') as f:
    #     inference_results = pk.load(f)
    with open('compute_time_results/baidu_FLIR_ref_results.defaultdict', 'rb') as f:
        ref_results = pk.load(f)
    with open('compute_time_results/baidu_FLIR_compress_results.defaultdict', 'rb') as f:
        compress_results = pk.load(f)
    with open('compute_time_results/baidu_FLIR_train_log.defaultdict', 'rb') as f:
        train_log = pk.load(f)
    with open('compute_time_results/baidu_FLIR_compress_results_inference.defaultdict', 'rb') as f:
        inference_results = pk.load(f)
    with open('compute_time_results/baidu_FLIR_ref_results_1000.defaultdict', 'rb') as f:
        ref_results_1000 = pk.load(f)
    # size
    ref_size_avg  = compute_att(ref_results,"ref_size")
    compress_size_avg = compute_att(compress_results,"compress_size")
    inference_size_avg = compute_att(inference_results,"compress_size")
    ref_1000_size_avg = compute_att(ref_results_1000,"compress_size")
    # acc
    paths = [i[0] for i in list(ref_results.items())]
    acc_total = 0
    for path in paths:
        ref_label = ref_results[path]['ref_label']
        results = inference_results[path]['results']
        if not ref_label in [line['keyword'] for line in results]:
            acc_reward = 0
        else:
            acc_reward = 1
        acc_total += acc_reward
    print("avg accuracy is %.5f" % (acc_total/len(paths)))
    # check error_code
    # error_code1 = compute_att(ref_results,"error_code")
    # error_code2 = compute_att(inference_results,"error_code")
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')
    # log
    # train_acc_r = train_log["acc_r"][-30:]
    # mean_acc = np.mean(train_log["acc_r"][-100:])
    # train_reward = train_log["reward"][-30:]
    # train_epsilon = train_log["epsilon"][-30:]

