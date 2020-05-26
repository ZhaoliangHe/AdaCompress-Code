import os
import pickle
from io import BytesIO
import time

from collections import defaultdict
import tensorflow as tf
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from keras.utils.np_utils import to_categorical

from imagenet import imagenet_label2class
from jpeg_utils import *

tf.set_random_seed(2)
np.random.seed(2)


def list_split(l, size):
    return [l[m:m + size] for m in range(0, len(l), size)]


class Environment(object):
    def __init__(self,
                 imagenet_train_path,
                 samples_per_class,
                 backbone_model,
                 backbone_model_input_size):

        self.imagenet_train_path = imagenet_train_path
        self.samples_per_class = samples_per_class

        self.deep_model = backbone_model
        self.deep_model.compile(optimizer="Adam", loss="categorical_crossentropy",
                                metrics=['top_k_categorical_accuracy'])
        self.deep_model_input_size = backbone_model_input_size

        self.image_datalist = []
        self.label_datalist = []

        self.ref_size_list = []
        self.ref_softlabels = []

        self.data_initial()
        self.gen_ref()
        self.reset()

    def _gen_sample_set(self):
        image_paths = []
        image_labels = []

        img_classes = os.listdir(self.imagenet_train_path)
        for img_class in img_classes:
            for image_name in np.random.choice(os.listdir("%s/%s" % (self.imagenet_train_path, img_class)),
                                               size=self.samples_per_class):
                sample_image_path = ("%s/%s/%s" % (self.imagenet_train_path, img_class, image_name))
                image_label = imagenet_label2class[image_name.split('_')[0]]

                image_paths.append(sample_image_path)
                image_labels.append(image_label)
        return image_paths, image_labels

    def data_initial(self):
        image_paths, image_labels = self._gen_sample_set()
        for image_path in image_paths:
            self.image_datalist.append(Image.open(image_path).convert("RGB"))
        self.label_datalist = to_categorical(image_labels, 1000)

    def gen_ref(self):
        ref_image_datas = []
        for idx, image in enumerate(self.image_datalist):
            ref_image_data, ref_size = self.process_single_image(image, 100)
            self.ref_size_list.append(ref_size)
            ref_image_datas.append(ref_image_data)

        self.ref_softlabels = self.deep_model.predict(np.array(ref_image_datas), batch_size=128, verbose=1)
        print("Reference soft labels generated..")

    def reset(self):
        self.curr_image_id = 0
        return self.image_datalist[self.curr_image_id]

    def process_single_image(self, image, quality):
        f = BytesIO()
        image.save(f, format='JPEG', quality=quality)
        size = len(f.getvalue())
        rec_img = Image.open(f)
        return preprocess_input(np.asarray(rec_img.resize(self.deep_model_input_size), dtype=np.float32)), size

    def model_recognize(self, image, quality, gt_label, ref_softlabel, ref_size):
        images_data_in, size = self.process_single_image(image, quality)
        soft_label = self.deep_model.predict(np.expand_dims(images_data_in, axis=0))[0]

        confidence = np.dot(soft_label, gt_label)
        ref_pod = np.dot(ref_softlabel, gt_label)

        acc_reward = np.clip(confidence / ref_pod, 0, 1)
        size_reward = size / ref_size
        return acc_reward, size_reward

    def step(self, action):
        done_flag = False

        info = {}

        quality = int(action)

        acc_reward, size_reward = self.model_recognize(image=self.image_datalist[self.curr_image_id],
                                                       quality=quality,
                                                       gt_label=self.label_datalist[self.curr_image_id],
                                                       ref_softlabel=self.ref_softlabels[self.curr_image_id],
                                                       ref_size=self.ref_size_list[self.curr_image_id])

        info['acc_r'] = acc_reward
        info['size_r'] = size_reward

        reward = acc_reward - size_reward

        self.curr_image_id += 1

        if self.curr_image_id >= len(self.image_datalist):
            done_flag = True
            return np.zeros(64), reward, done_flag, info

        features = self.image_datalist[self.curr_image_id]
        return features, reward, done_flag, info


class EnvironmentAPI(object):
    def __init__(self,
                 imagenet_train_path,
                 samples_per_class,
                 cloud_agent,
                 reference_quality=95):

        self.imagenet_train_path = imagenet_train_path
        self.samples_per_class = samples_per_class
        self.cloud_agent = cloud_agent

        self.image_datalist = []
        self.label_datalist = []
        self.image_paths = []

        self.references = defaultdict(dict)
        self.ref_size_list = []
        self.ref_labels = []
        self.ref_confidences = []

        self.data_initial()
        self.gen_ref(reference_quality)
        self.reset()

    def _gen_sample_set(self):
        image_paths = []
        image_labels = []

        img_classes = os.listdir(self.imagenet_train_path)
        for img_class in img_classes:
            for image_name in np.random.choice(os.listdir("%s/%s" % (self.imagenet_train_path, img_class)),
                                               size=self.samples_per_class):
                sample_image_path = ("%s/%s/%s" % (self.imagenet_train_path, img_class, image_name))
                image_label = imagenet_label2class[image_name.split('_')[0]]

                image_paths.append(sample_image_path)
                image_labels.append(image_label)
        return image_paths, image_labels

    def data_initial(self):
        image_paths, image_labels = self._gen_sample_set()
        for image_path in image_paths:
            self.image_datalist.append(Image.open(image_path).convert("RGB"))
            self.image_paths.append(image_path)

    def gen_ref(self, ref_quality):
        if not os.path.exists("evaluation_results/%s_ref.pkl" % self.cloud_agent.api_name):
            print("Reference not exists, generating...")
            for idx, image in enumerate(self.image_datalist):
                img_path = self.image_paths[idx]
                time.sleep(0.1)
                if idx % 20 == 0:
                    print(".", end='')
                error_code, reg_results, ref_size = self.cloud_agent.recognize(image, ref_quality)
                if error_code == 0:
                    gt_id = np.argmax([line['score'] for line in reg_results])

                    self.references[img_path]['error_code'] = error_code
                    self.references[img_path]['ref_size'] = ref_size
                    self.references[img_path]['ref_label'] = reg_results[gt_id]['keyword']
                    self.references[img_path]['ref_confidence'] = reg_results[gt_id]['score']
                else:
                    self.references[img_path]['error_code'] = error_code
                    self.references[img_path]['error_msg'] = reg_results
                    self.references[img_path]['ref_size'] = ref_size

            with open("evaluation_results/%s_ref.pkl" % self.cloud_agent.api_name, 'wb') as f:
                pickle.dump(self.references, f)
            print("\nReference generated..")
        else:
            with open("evaluation_results/%s_ref.pkl" % self.cloud_agent.api_name, 'rb') as f:
                self.references = pickle.load(f)
            print("Reference loaded...")

    def reset(self):
        self.curr_image_id = 0
        return self.image_datalist[self.curr_image_id]

    def cloud_recognize(self, image, quality, gt_label, ref_confidence, ref_size):
        error_code, reg_results, size = self.cloud_agent.recognize(image, quality)
        size_reward = size / ref_size
        if error_code == 0:
            if not gt_label in [line['keyword'] for line in reg_results]:
                return 0, 0, size_reward
            else:
                reg_id = [line['keyword'] for line in reg_results].index(gt_label)
                confidence = np.clip([line['score'] for line in reg_results][reg_id] / ref_confidence, 0, 1)
                acc_reward = confidence

                return 0, acc_reward, size_reward
        else:
            return 1, reg_results[0], 0

    def step(self, action):
        done_flag = False

        info = {}

        quality = int(action)

        reference = self.references[self.image_paths[self.curr_image_id]]
        if reference['error_code'] == 0:
            error_code, acc_reward, size_reward = self.cloud_recognize(image=self.image_datalist[self.curr_image_id],
                                                                       quality=quality,
                                                                       gt_label=reference['ref_label'],
                                                                       ref_confidence=reference['ref_confidence'],
                                                                       ref_size=reference['ref_size']
                                                                       )
            if error_code == 0:

                reward = acc_reward - size_reward

                info['acc_r'] = acc_reward
                info['size_r'] = size_reward
                info['action'] = action
                info['reward'] = reward

                self.curr_image_id += 1

                if self.curr_image_id >= len(self.image_datalist):
                    done_flag = True
                    return 0, np.zeros(64), reward, done_flag, info

                features = self.image_datalist[self.curr_image_id]
                return 0, features, reward, done_flag, info

            else:
                self.curr_image_id += 1
                return 1, None, None, None, None
        else:
            self.curr_image_id += 1
            return 2, None, None, None, None
