# encoding: utf-8
import tensorflow as tf
from res_manager import ResultManager
import numpy as np
from collections import defaultdict

from keras.applications.mobilenetv2 import preprocess_input
from keras.applications import InceptionV3

from Agents import Q_Agent
from keras.models import load_model
from ImageCompressionEnvironment import BatchImgEnvironment

np.set_printoptions(precision=3)

tf.set_random_seed(2)
np.random.seed(2)

BATCH_SIZE = 128

def list_split(l, size):
    return [l[m:m + size] for m in range(0, len(l), size)]


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
                              samples_per_class=3,
                              backbone_model=InceptionV3(),
                              backbone_model_input_size=(299, 299))

    env.deep_model.compile(optimizer="Adam", loss="categorical_crossentropy",
                            metrics=['top_k_categorical_accuracy'])

    env.reset()
    image_data_batchs = list_split(env.image_datalist, BATCH_SIZE)
    image_label_batchs = list_split(env.label_datalist, BATCH_SIZE)
    ref_size_batchs = list_split(env.ref_size_list, BATCH_SIZE)


    evaluation_result = defaultdict(list)

    for idx, image_batch in enumerate(image_data_batchs):
        performance = defaultdict(list)
        print("\n\nbatch %s:" % (idx + 1))

        curr_label_batch = image_label_batchs[idx]
        curr_bm_size_batch = ref_size_batchs[idx]

        acc_list = []
        size_list = []
        banchmark_size = []

        for img_id, image in enumerate(image_batch):
            label = curr_label_batch[img_id]

            image_data = preprocess_input(np.expand_dims(np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0))
            features = np.array([0, 1]) if np.argmax(feature_extractor.predict(image_data)) == 1 else np.array([1, 0])

            step_count += 1
            # action_id = agent.choose_action(features)
            action_id = 2 if np.argmax(features) == 0 else 5
            action = [i for i in np.arange(5, 105, 10)][action_id]

            processed_img, size = env.process_single_image(image, int(action))
            _, ref_size = env.process_single_image(image, 75)

            accuracy = env.deep_model.evaluate(x=np.expand_dims(processed_img, axis=0), y=np.expand_dims(label, axis=0), verbose=2)[1]
            acc_list.append(accuracy)
            size_list.append(size)
            banchmark_size.append(ref_size)

        batch_acc = np.sum(acc_list) / len(acc_list)
        batch_aver_size = np.mean(size_list)

        print("batch %s\taccuray %.2f\taverage size %.2f" % (idx+1, batch_acc, batch_aver_size))
        evaluation_result['accuracies'].append(batch_acc)
        evaluation_result['batch_sizes'].append(batch_aver_size)
        evaluation_result['sizes'] += size_list
        evaluation_result['bm_sizes'] += banchmark_size

    # except Exception as e:
    #     print(e)
    #     print("Exit...")
    # finally:
    #     print("saving...")
    #     if not EVALUATION:
    #         agent.save_params("Agent_params/")
    rm.save(evaluation_result,
            name='Q_agent_eval',
            topic='AgentTest',
            comment="evaluation result on 3K of Q agent(ref Q=75)",
            # replace_version='latest',
            )
