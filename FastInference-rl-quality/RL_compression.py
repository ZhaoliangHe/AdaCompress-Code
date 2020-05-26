# encoding: utf-8
import tensorflow as tf
from res_manager import ResultManager
import numpy as np
from collections import defaultdict

from keras.applications.mobilenetv2 import preprocess_input
from keras.applications import InceptionV3

from Agents import DDPG_Agent, Q_Agent
from keras.models import load_model
from ImageCompressionEnvironment import BatchImgEnvironment

np.set_printoptions(precision=3)

tf.set_random_seed(2)
np.random.seed(2)

EVALUATION = True

if __name__ == '__main__':
    images_dir = '/home/hsli/gnode02/imagenet-data/train/'

    feature_extractor = load_model(
        'checkpoints/mobilenetv2_predictor_2W_acc_0.6955_epoch50.hdf5')

    rm = ResultManager('results')
    agent_acc_size_dict = []
    origin_acc_size_dict = []

    agent = Q_Agent(s_dim=2,
                    a_dim=10,
                    epsilon_decay=0.9993,
                    epsilon_min=0.2,
                    lr=0.1,
                    gamma=0.95)

    step_count = 0

    env = BatchImgEnvironment(imagenet_train_path=images_dir,
                              samples_per_class=2,
                              backbone_model=InceptionV3(),
                              backbone_model_input_size=(299, 299))

    # try:
    for i_episode in range(20):
        performance = defaultdict(list)
        print("\n\nepisode %s:" % i_episode)
        image = env.reset()

        image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
        features = np.array([0, 1]) if np.argmax(feature_extractor.predict(image_data)) == 1 else np.array([1, 0])

        while True:
            step_count += 1
            # action_id = agent.choose_action(features)
            action_id = 2 if np.argmax(features) == 0 else 5
            action = [i for i in np.arange(5, 105, 10)][action_id]


            new_image, reward, done_flag, info = env.step(action)

            performance['acc_r'].append(info['acc_r'])
            performance['size_r'].append(info['size_r'])
            performance['action'].append(action)
            performance['reward'].append(reward)

            if not done_flag:
                image_data = preprocess_input(
                    np.expand_dims(np.asarray(new_image.resize((224, 224)), dtype=np.float32), axis=0))
                new_features = np.array([0, 1]) if np.argmax(feature_extractor.predict(image_data)) == 1 else np.array(
                    [1, 0])
                if not EVALUATION:
                    agent.learn(features, action_id, reward, new_features)
            else:
                break

            features = new_features

        for k, v in performance.items():
            print("%s: %.3f" % (k, np.mean(v)), end='\t')
        print("\n", agent.q_table)
        print(agent.curr_exploration_rate)
        print(np.argmax(agent.q_table, axis=1))

    # except Exception as e:
    #     print(e)
    #     print("Exit...")
    # finally:
    #     print("saving...")
    #     if not EVALUATION:
    #         agent.save_params("Agent_params/")
    # rm.save(performance,
    #         name='reward_test',
    #         topic='AgentTest',
    #         comment="mse and size statistics",
    #         replace_version='latest',
    #         )
