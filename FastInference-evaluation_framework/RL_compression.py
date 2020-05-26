# encoding: utf-8
import tensorflow as tf
import time
from res_manager import ResultManager
import numpy as np
from collections import defaultdict, deque

from keras.applications.mobilenetv2 import preprocess_input

from Agents import Q_Agent
from keras.models import load_model
from ImageCompressionEnvironment import EnvironmentAPI
from cloud_apis import Baidu, FacePP

import matplotlib.pyplot as plt

plt.ion()

np.set_printoptions(precision=3)

tf.set_random_seed(2)
np.random.seed(2)

EVALUATION = False


def plot_durations(y):
    plt.figure(2)
    plt.clf()
    plt.subplot(511)
    plt.plot(y[:, 0])
    plt.ylabel('confidence')
    plt.subplot(512)
    plt.plot(y[:, 1])
    plt.ylabel('compression rate')
    plt.subplot(513)
    plt.plot(y[:, 2])
    plt.ylabel('reward')
    plt.subplot(514)
    plt.plot(y[:, 3])
    plt.ylabel('epsilon')
    plt.subplot(515)
    plt.plot(y[:, 4])
    plt.ylabel('action')
    plt.pause(0.0001)


if __name__ == '__main__':
    images_dir = '/home/hsli/imagenet-data/train/'

    feature_extractor = load_model(
        'checkpoints/mobilenetv2_predictor_2W_acc_0.6955_epoch50.hdf5')

    rm = ResultManager('results')
    agent_acc_size_dict = []
    origin_acc_size_dict = []

    agent = Q_Agent(s_dim=2,
                    a_dim=10,
                    epsilon_decay=0.991,
                    epsilon_min=0.02,
                    lr=0.4,
                    gamma=0.92)

    step_count = 0

    env = EnvironmentAPI(imagenet_train_path=images_dir,
                         samples_per_class=37,
                         cloud_agent=FacePP())

    # try:
    performance = defaultdict(list)
    plot_y = []
    plot_part = deque(maxlen=5)

    for i_episode in range(2):
        print("\n\nepisode %s:" % i_episode)
        image = env.reset()

        image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
        features = np.array([0, 1]) if np.argmax(feature_extractor.predict(image_data)) == 1 else np.array([1, 0])

        while True:
            step_count += 1
            action_id = agent.choose_action(features)
            action = [i for i in np.arange(5, 105, 10)][action_id]

            error_code, new_image, reward, done_flag, info = env.step(action)

            if error_code > 0:
                print(error_code)
                continue

            time.sleep(0.1)

            performance['acc_r'].append(info['acc_r'])
            performance['size_r'].append(info['size_r'])
            performance['action'].append(action)
            performance['reward'].append(reward)
            performance['epsilon'].append(agent.curr_exploration_rate)

            print('\tstep %d\t' % step_count, end='\t')
            for k, v in info.items():
                print("%s: %.3f" % (k, v), end='\t')
            print('\n')

            if not done_flag:
                image_data = preprocess_input(
                    np.expand_dims(np.asarray(new_image.resize((224, 224)), dtype=np.float32), axis=0))
                new_features = np.array([0, 1]) if np.argmax(feature_extractor.predict(image_data)) == 1 else np.array(
                    [1, 0])
                if not EVALUATION:
                    agent.learn(features, action_id, reward, new_features)
                    if step_count <= 100:
                        agent.curr_exploration_rate = 1
            else:
                break

            plot_part.append(np.array([info['acc_r'], info['size_r'], reward, agent.curr_exploration_rate, action]))
            if step_count % 5 == 0:
                plot_y.append(np.mean(plot_part, axis=0))
                plot_durations(np.array(plot_y))

            features = new_features

            if step_count % 20 == 0:
                print(agent.q_table)
                print(np.argmax(agent.q_table, axis=1))

            if step_count >= 380:
                break

        rm.save(performance,
                name='training_log',
                topic="AgentTrain",
                comment="Baidu environment's training log",
                # replace_version='latest'
                )
