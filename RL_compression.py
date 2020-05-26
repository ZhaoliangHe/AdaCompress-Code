# encoding: utf-8
import os
from res_manager import ResultManager
import numpy as np
from collections import defaultdict

from keras.applications import MobileNetV2

from Agents import PG_Agent, DDPG_Agent
from ImageCompressionEnvironment import BatchImgEnvironment

np.random.seed(2)

if __name__ == '__main__':
    model = MobileNetV2()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    images_dir = '/home/hsli/imagenet-data/train/'

    rm = ResultManager('results')
    agent_acc_size_dict = []
    origin_acc_size_dict = []

    agent = DDPG_Agent(a_dim=64,
                       s_dim=64,
                       train_batchsize=128)

    epsilon = 0.8  # control exploration
    epsilon_decay = 0.99

    env = BatchImgEnvironment(imagenet_train_path=images_dir,
                              samples_per_class=1024,
                              step_batch_size=128,
                              deep_model=model)

    try:

        accuracies = defaultdict(list)
        sizes = defaultdict(list)
        count = 0

        for i_episode in range(5000):
            print("\n\nepisode %s:" % i_episode)
            observation = env.reset()

            while True:

                count += 1

                ref_action = np.vstack([[16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22,
                                         24, 49, 35, 37, 29, 40, 58,
                                         51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109,
                                         81, 87, 95, 98, 103, 104, 103,
                                         62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99]] * 128)

                choise_idx = np.random.choice([0, 1], p=[epsilon, 1 - epsilon])
                epsilon *= epsilon_decay

                if choise_idx == 0:
                    action = ref_action
                else:
                    action = agent.choose_action(observation)
                    action = np.random.normal(action, 0.1)

                new_observation, reward, done_flag, info = env.step(action)

                print("\tchoise %d\tepsilon %.2f\treward %.2f\taccuracy %.2f\taver size %d" % (
                    choise_idx, epsilon, reward, info['accuracy'], info['average size']))

                if choise_idx == 1:
                    accuracies['agent'].append(info['accuracy'])
                    sizes['agent'].append(info['average size'])
                    accuracies['agent_id'].append(count)
                    sizes['agent_id'].append(count)
                else:
                    accuracies['origin'].append(info['accuracy'])
                    sizes['origin'].append(info['average size'])
                    accuracies['origin_id'].append(count)
                    sizes['origin_id'].append(count)

                if not done_flag:
                    agent.store_transition(observation, action, reward, new_observation)

                if agent.memory.index >= 3 * 64:
                    print("\ttraining...")
                    agent.learn(epochs=10)

                if done_flag:
                    agent.learn(epochs=20)
                    break

                observation = new_observation


    except KeyboardInterrupt as e:
        print("Exit...")
    finally:
        print("saving...")
        rm.save(accuracies, name="accuracies", topic="AgentTest", comment="all accuracies info")
        rm.save(sizes, name="sizes", topic="AgentTest", comment="all sizes info")
