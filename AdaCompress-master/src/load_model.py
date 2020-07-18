from collections import defaultdict, deque
from PIL import Image

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from src.agents import DQN_Agent
from keras.applications import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input
from keras.layers import AveragePooling2D
from keras.models import load_model, Model

# from src.cloud_apis import AmazonRekognition
# from src.cloud_apis import FacePP
from src.cloud_apis import Baidu
import time
import multiprocessing
from multiprocessing import Process
import os

def _gen_sample_set_imagenet(imagenet_train_path, samples_per_class):
    image_paths = []
    img_classes = os.listdir(imagenet_train_path)
    for img_class in img_classes:
        for image_name in np.random.choice(os.listdir("%s/%s" % (imagenet_train_path, img_class)),
                                           size=samples_per_class):
            sample_image_path = ("%s/%s/%s" % (imagenet_train_path, img_class, image_name))

            image_paths.append(sample_image_path)
    return image_paths
imagenet_paths = _gen_sample_set_imagenet('/home/imagenet-data/train/', 2)
#MobileNetV2
feature_extractor = MobileNetV2(include_top=False)
x = feature_extractor.output
x = AveragePooling2D(pool_size=(4, 4))(x)
feature_extractor = Model(inputs=feature_extractor.input, outputs=x)
# agent
agent = DQN_Agent(s_dim=1280,
                  a_dim=10,
                  epsilon_decay=0.99,
                  epsilon_min=0.02,
                  gamma=0.95,
                  replay_batchsize=256)

def run_proc(arg):
    """子进程要执行的代码"""
    # load_model_time_start = time.time()
    agent.model = load_model('compute_time_results/baidu_imagenet.h5')
    arg = agent.model.get_weights()
    arg.extend([1,2,3,4])
    print('子进程运行中，pid=%d...' % os.getpid())  # os.getpid获取当前进程的进程号
    print('子进程将要结束...')
    # load_model_time_end = time.time()
    # print("load model time %.5f" % (load_model_time_end-load_model_time_start) )

if __name__ == '__main__':
    array = multiprocessing.Manager().list()
    # test_image_paths = imagenet_paths[:5]
    # load_model_time_start = time.time()
    # # agent.model = load_model('compute_time_results/baidu_FLIR.h5')
    # load_model_time_end = time.time()
    # print("load model time %.5f" % (load_model_time_end-load_model_time_start) )
    print('父进程pid: %d' % os.getpid())  # os.getpid获取当前进程的进程号
    p = Process(target=run_proc, args=(array,))
    # load_model_time_start = time.time()
    p.start()
    # agent.model = load_model('compute_time_results/baidu_imagenet.h5')
    # load_model_time_end1 = time.time()
    # print("load model time %.5f" % (load_model_time_end1-load_model_time_end) )
    # time.sleep(3)
    # agent.model.load_weights(array)
    # image = Image.open(test_image_paths[0]).convert("RGB")
    # image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
    # features = feature_extractor.predict(image_data)[0][0][0]
    # state_actions, action_id = agent.choose_action(features)
    # action = [i for i in np.arange(5, 105, 10)][action_id]
    # quality = int(action)
    # p.join()
    # p.kill()
    # load_model_time_end = time.time()
    # print("load model time %.5f" % (load_model_time_end-load_model_time_start) )

# if __name__ == '__main__':
#     main()