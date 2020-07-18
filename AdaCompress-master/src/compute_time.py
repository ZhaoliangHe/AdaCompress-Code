# encoding: utf-8
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
import os
import psutil

process = psutil.Process(os.getpid())
print(98, 'ID: ', os.getpid(), 'Used Memory:', (process.memory_info().rss - process.memory_info().shared) / 1024 / 1024, 'MB')

np.set_printoptions(precision=3)
tf.set_random_seed(2)
np.random.seed(2)

# EVALUATION = False
EVALUATION = True

def compute_memory(step_cout):
    process = psutil.Process(os.getpid())
    print(step_cout,'ID: ',os.getpid(), 'Used Memory:', (process.memory_info().rss - process.memory_info().shared) / 1024 / 1024, 'MB')

# dataset
FLIR = []
for time_stemp in os.listdir("/home/hezhaoliang/dataset/FLIR/train/thermal_8_bit/"):
    FLIR.append("/home/hezhaoliang/dataset/FLIR/train/thermal_8_bit/%s" % (
        time_stemp))

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

# MobileNetV2
feature_extractor = MobileNetV2(include_top=False)
x = feature_extractor.output
x = AveragePooling2D(pool_size=(4, 4))(x)
feature_extractor = Model(inputs=feature_extractor.input, outputs=x)
compute_memory(101)
# agent
agent = DQN_Agent(s_dim=1280,
                  a_dim=10,
                  epsilon_decay=0.99,
                  epsilon_min=0.02,
                  gamma=0.95,
                  replay_batchsize=256)

if __name__ == '__main__':
    test_image_paths = imagenet_paths[:5] # FLIR[:1000]
    # test_image_paths = FLIR[:1000]
    cloud_agent = Baidu()

    train_log = defaultdict(list)
    ref_results = defaultdict(dict)
    compress_results = defaultdict(dict)

    choose_action_total_time = 0
    feedback_total_time = 0
    feature_total_time = 0
    recent_accuracy = 0
    recent_reward = 0

    start_time = time.time()
    if EVALUATION:
        agent.model = load_model('compute_time_results/baidu_imagenet.h5')
        # agent.model = load_model('compute_time_results/baidu_FLIR.h5')
        agent.curr_exploration_rate = 0
    load_model_time = time.time()
    # init feature
    compute_memory(103)
    image = Image.open(test_image_paths[0]).convert("RGB")
    image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
    features = feature_extractor.predict(image_data)[0][0][0]
    compute_memory(104)
    # for
    for step_count, path in enumerate(test_image_paths):
        compute_memory(step_count)
        image = Image.open(path).convert("RGB")
        raw_image_time = time.time()
        image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
        new_features = feature_extractor.predict(image_data)[0][0][0]
        feature_time = time.time()
        feature_once_time = feature_time - raw_image_time
        # print("get feature time: ", feature_once_time)
        feature_total_time += feature_once_time
        # choose action
        raw_feature_time = time.time()
        state_actions, action_id = agent.choose_action(new_features)
        action = [i for i in np.arange(5, 105, 10)][action_id]
        quality = int(action)
        choose_action_time = time.time()
        choose_action_once_time = choose_action_time - raw_feature_time
        # print("choose action once time: ",choose_action_once_time)
        choose_action_total_time += choose_action_once_time
        # quality = 75
        # recognize
        error_code2, results, compress_size = cloud_agent.recognize(image, quality)
        feedback_time = time.time()
        feedback_once_time = feedback_time - choose_action_time
        print("upload and feedback time: ", feedback_once_time)
        feedback_total_time += feedback_once_time
        if not EVALUATION:
            error_code1, reg_results, ref_size = cloud_agent.recognize(image, quality=75)
            gt_id = np.argmax([line['score'] for line in reg_results])
            ref_label = reg_results[gt_id]['keyword'] # ground truth
            feedback_time = time.time()
            feedback_once_time = feedback_time - choose_action_time
            # print("upload and feedback time: ", feedback_once_time)
            feedback_total_time += feedback_once_time
            if error_code1 == 0 and error_code2 ==0:
                if not ref_label in [line['keyword'] for line in results]:
                    acc_reward = 0
                else:
                    acc_reward = 1
                size_reward = compress_size / ref_size
                reward = acc_reward - size_reward
            else:
                continue
        # cache
        if not EVALUATION:
            ref_results[path]['error_code'] = error_code1
            ref_results[path]['ref_size'] = ref_size
            ref_results[path]['ref_label'] = reg_results[gt_id]['keyword']
            ref_results[path]['ref_confidence'] = reg_results[gt_id]['score']
            compress_results[path]['error_code'] = error_code2
            compress_results[path]['compress_size'] = compress_size
            compress_results[path]['quality'] = quality
            compress_results[path]['results'] = results
            train_log['image_path'].append(path)
            train_log['acc_r'].append(acc_reward)
            train_log['size_r'].append(size_reward)
            train_log['action'].append(action)
            train_log['reward'].append(reward)
            train_log['epsilon'].append(agent.curr_exploration_rate)
            # recent
            if step_count > 10:
                recent_accuracy = np.mean(train_log['acc_r'][-10:])
                recent_reward = np.mean(train_log['reward'][-10:])
        else:
            compress_results[path]['error_code'] = error_code2
            compress_results[path]['compress_size'] = compress_size
            compress_results[path]['quality'] = quality
            compress_results[path]['results'] = results

        if not EVALUATION:
            agent.remember(features, action_id, reward, new_features)
            if 128 <= step_count <= 1600 and step_count % 5 == 0:
                agent.learn()
            if step_count <= 128:
                agent.curr_exploration_rate = 1
        features = new_features

        if step_count % 10 == 0:
            if not EVALUATION:
                # Update RL agent model
                # agent.model.save("compute_time_results/baidu_imagenet.h5")
                # with open('compute_time_results/baidu_ref_results.defaultdict', 'wb') as f:
                #     pickle.dump(ref_results, f)
                # with open('compute_time_results/baidu_compress_results.defaultdict', 'wb') as f:
                #     pickle.dump(compress_results, f)
                # with open('compute_time_results/baidu_train_log.defaultdict', 'wb') as f:
                #     pickle.dump(train_log, f)
                agent.model.save("compute_time_results/baidu_FLIR.h5")
                with open('compute_time_results/baidu_FLIR_ref_results.defaultdict', 'wb') as f:
                    pickle.dump(ref_results, f)
                with open('compute_time_results/baidu_FLIR_compress_results.defaultdict', 'wb') as f:
                    pickle.dump(compress_results, f)
                with open('compute_time_results/baidu_FLIR_train_log.defaultdict', 'wb') as f:
                    pickle.dump(train_log, f)
            else:
                # with open('compute_time_results/baidu_FLIR_compress_results_inference.defaultdict', 'wb') as f:
                # with open('compute_time_results/baidu_FLIR_ref_results_1000.defaultdict', 'wb') as f:
                with open('compute_time_results/baidu_ImageNet_test1', 'wb') as f:
                    pickle.dump(compress_results, f)


        if recent_reward > 0.45 and recent_accuracy > 0.8 and agent.curr_exploration_rate < 0.4 \
            and not EVALUATION:
            end_time = time.time()
            train_time = end_time - start_time
            print("done")
            agent.model.save("compute_time_results/baidu_imagenet.h5")
            print("compete ,using %d images, the train time is %.5f" % (step_count, train_time))
            break
        compute_memory(step_count)
    print("load model time: ",load_model_time - start_time)
    print("feature %.5f" % (feature_total_time/step_count))
    print("choose action %.5f" % (choose_action_total_time/step_count))
    print("feedback %.5f" % (feedback_total_time/step_count))