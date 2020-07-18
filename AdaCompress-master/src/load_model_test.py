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
# feature_extractor = MobileNetV2(include_top=False)
# x = feature_extractor.output
# x = AveragePooling2D(pool_size=(4, 4))(x)
# feature_extractor = Model(inputs=feature_extractor.input, outputs=x)
# agent
agent = DQN_Agent(s_dim=1280,
                  a_dim=10,
                  epsilon_decay=0.99,
                  epsilon_min=0.02,
                  gamma=0.95,
                  replay_batchsize=256)
agent1 = DQN_Agent(s_dim=1280,
                  a_dim=10,
                  epsilon_decay=0.99,
                  epsilon_min=0.02,
                  gamma=0.95,
                  replay_batchsize=256)
agent2 = DQN_Agent(s_dim=1280,
                  a_dim=10,
                  epsilon_decay=0.99,
                  epsilon_min=0.02,
                  gamma=0.95,
                  replay_batchsize=256)
agent3 = DQN_Agent(s_dim=1280,
                  a_dim=10,
                  epsilon_decay=0.99,
                  epsilon_min=0.02,
                  gamma=0.95,
                  replay_batchsize=256)
def process_handle(arg):
    arg1 = load_model('compute_time_results/baidu_imagenet.h5')
    arg.append(arg1)
    # arg.extend([1, 2, 3, 4])
    print('子进程运行中，pid=%d...' % os.getpid())  # os.getpid获取当前进程的进程号
    print('子进程将要结束...')
    print(arg)

def process_handle1(arg):
    arg1 = load_model('compute_time_results/baidu_imagenet.h5')
    arg.append(arg1)
    print('子进程运行中，pid=%d...' % os.getpid())  # os.getpid获取当前进程的进程号
    print('子进程将要结束...')
    print(arg)

if __name__ == "__main__":
# def main():
#     model_w = agent.model.get_weights()
#     agent.model = load_model('evaluation_results/agent_DQN_baidu_FLIR.h5')
    array = multiprocessing.Manager().list()
    # array1 = multiprocessing.Manager().list()
    print('父进程pid: %d' % os.getpid())  # os.getpid获取当前进程的进程号
    handle_process = Process(target=process_handle, args=(array,))
    handle_process.start()
    print("xx",array)
    # handle_process.join()
    # print(array)
    # agent.model = array[0]
    # print("agent model", agent.model.get_weights())
    # handle_process1 = Process(target=process_handle1, args=(array1,))
    # handle_process1.start()
    # handle_process.join()
    # agent.model = load_model('evaluation_results/agent_DQN_baidu_FLIR.h5')
    # print("FLIR model", agent.model.get_weights())
    test_image_paths = imagenet_paths[:10]
    print(test_image_paths)
    flag = 1
    for index,path in enumerate(test_image_paths):
        image = Image.open(path).convert("RGB")
        image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
        # features = feature_extractor.predict(image_data)[0][0][0]
        # state_actions, action_id = agent.choose_action(features)
        # action = [i for i in np.arange(5, 105, 10)][action_id]
        print(index)
        print(array)
        # time.sleep(1)
        if flag==1:
            print("xxx")
            print(array)
            handle_process.join()
            agent.model = array[0]
            flag = 0

# if __name__ == "__main__":
#     main()
    # array = multiprocessing.Manager().list()
    # print('父进程pid: %d' % os.getpid())  # os.getpid获取当前进程的进程号
    # handle_process = Process(target=process_handle, args=(array,))
    # handle_process.start()
    # # handle_process.join()
    # array.extend([5, 6, 7])
    # print(array)