import cv2
from PIL import Image
import pickle
from imagenet import imagenet_label2class
from functools import partial
import os
from jpeg_utils import *
from io import BytesIO
from pathos.multiprocessing import ProcessingPool as Pool

from keras.applications.mobilenetv2 import preprocess_input
from keras.utils.np_utils import to_categorical

model_input_size = (224, 224)


def list_split(l, size):
    return [l[m:m + size] for m in range(0, len(l), size)]


class BatchImgEnvironment(object):
    def __init__(self,
                 imagenet_train_path,
                 samples_per_class,
                 step_batch_size,
                 deep_model,
                 q_block_shape=(8, 8)):

        self.imagenet_train_path = imagenet_train_path
        self.deep_model = deep_model
        self.q_block_shape = q_block_shape
        self.samples_per_class = samples_per_class
        self.batch_size = step_batch_size

        self.image_data_batches = None

        self.train_image_batches = None
        self.train_label_batches = None

        self.curr_batch_id = 0

        self.data_initial()
        self.reset()

    def gen_sample_set(self):
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
        image_paths, image_labels = self.gen_sample_set()
        image_datalist = []
        for image_path in image_paths:
            image_datalist.append(Image.open(image_path).convert("RGB"))
        training_label = to_categorical(image_labels, 1000)

        self.image_data_batches = list_split(image_datalist, self.batch_size)
        self.train_label_batches = list_split(training_label, self.batch_size)

    def reset(self):
        self.curr_batch_id = 0
        return self.get_batch_features()

    def get_single_features(self, image):
        y, u, v = image.split()
        channel = np.asarray(y.resize((512, 512)))
        # split and shift
        blocks = split_88(channel) - 128
        # DCT
        dct_blocks = np.array([cv2.dct(block) for block in blocks])

        # --- Extract features --- #
        dct_std_matrix = np.std(dct_blocks, axis=0, ddof=1)

        return dct_std_matrix.flatten()

    def get_batch_features(self):
        # pool = Pool(6)
        # batch_features = pool.map(self.get_single_features, self.image_data_batches[self.curr_batch_id])
        batch_features = []
        for image in self.image_data_batches[self.curr_batch_id]:
            batch_features.append(self.get_single_features(image))
        return np.array(batch_features)

    def process_batch_image(self, actions):
        processed_images_sizes = []
        for idx, action in enumerate(actions):
            image = self.image_data_batches[self.curr_batch_id][idx]
            q_table = 1. / action
            processed_images_sizes.append(self.process_single_image(image, q_table))
        batch_processed_images = np.array([np.asarray(line[0].resize(model_input_size), dtype=np.float32) for line in processed_images_sizes])
        batch_aver_size = np.mean([line[1] for line in processed_images_sizes])
        return batch_processed_images, batch_aver_size

    def process_single_image(self, image, q_table):
        pil_q_table = {0: q_table.astype(np.uint8).flatten().tolist()}
        f = BytesIO()
        image.save(f, format='JPEG', qtables=pil_q_table)
        size = len(f.getvalue())
        rec_img = Image.open(f)
        return [rec_img, size]

    def step(self, batch_actions):
        done_flag = False

        info = {}

        images_data_in, aver_size = self.process_batch_image(batch_actions)
        images_data_in = preprocess_input(images_data_in)
        loss, accuracy = self.deep_model.evaluate(images_data_in, self.train_label_batches[self.curr_batch_id], verbose=2)

        prediction_reward = accuracy
        compression_reward = 1 / aver_size

        info['accuracy'] = accuracy
        info['average size'] = aver_size

        reward = 2 ** (20 * prediction_reward)
        # reward = 1e7 / aver_size
        # reward = np.mean(batch_actions)

        self.curr_batch_id += 1

        if self.curr_batch_id >= len(self.train_label_batches):
            done_flag = True
            return np.zeros((self.batch_size, 64)), reward, done_flag, info

        features = self.get_batch_features()
        return features, reward, done_flag, info


class Environment(object):
    def __init__(self,
                 images_dir_path,
                 image_file_names,
                 deep_model,
                 q_block_shape=(8, 8)):

        self.images_dir_path = images_dir_path
        self.image_file_names = image_file_names
        self.deep_model = deep_model
        self.q_block_shape = q_block_shape

        self.training_image_file_names = []

        self.data_initial()
        self.reset()

    def data_initial(self):
        """
        Pick out those images that are robust to 'lossless' compression
        :return:
        """
        # for image_name in self.image_file_names:
        #     self.curr_image = Image.open("%s/%s" % (self.images_dir_path, image_name))
        #     if np.argmax(self.deep_model.predict(
        #             preprocess_input(np.expand_dims(self._process_single_image(np.ones((8, 8))), axis=0)))[0]) == 508 and np.argmax(self.deep_model.predict(preprocess_input(np.expand_dims(np.asarray(self.curr_image.convert("RGB").resize(model_input_size)).astype(np.float32), axis=0)))[0]) == 508:
        #         self.training_image_file_names.append(image_name)
        with open('training_image_names.list', 'rb') as f:
            self.training_image_file_names = pickle.load(f)[:300]

    def _sample_generator(self):
        while True:
            for image_filename in self.training_image_file_names:
                yield Image.open("%s/%s" % (self.images_dir_path, image_filename))

    def reset(self):
        self.generator = self._sample_generator()
        self.curr_image = next(self.generator)
        self.endurance = 0
        self.curr_reward = 0
        return self.get_features()

    def get_features(self):
        y, u, v = self.curr_image.convert("RGB").resize((224, 224)).convert("YCbCr").split()
        channel = np.asarray(y)
        # split and shift
        blocks = split_88(channel) - 128
        # DCT
        dct_blocks = np.array([cv2.dct(block) for block in blocks])

        # --- Extract features --- #
        dct_mean_matrix = np.mean(dct_blocks, axis=0)
        dct_std_matrix = np.std(dct_blocks, axis=0, ddof=1)

        return np.hstack([dct_mean_matrix.flatten(), dct_std_matrix.flatten()])

    def _process_single_image(self, q_table):
        y, u, v = self.curr_image.convert("RGB").resize((224, 224)).convert("YCbCr").split()

        channel = np.asarray(y)
        # split and shift
        blocks = split_88(channel) - 128
        # DCT
        dct_blocks = np.array([cv2.dct(block) for block in blocks])

        # --- These blocks are for storage and transmission--- #
        div_blocks = np.array([np.round(dct_block / q_table) for dct_block in dct_blocks])

        scale_blocks = np.array([div_block * q_table for div_block in div_blocks])
        idct_blocks = np.array([cv2.idct(rec_block) for rec_block in scale_blocks])
        rec_blocks = np.clip(idct_blocks + 128, 0, 255)
        rec_channel = merge_88(rec_blocks.astype(np.uint8))

        # _u = u.resize((112, 112)).resize((224, 224))
        # _v = v.resize((112, 112)).resize((224, 224))

        _u = u
        _v = v

        rec_image = Image.merge("YCbCr", [Image.fromarray(rec_channel), _u, _v])
        rec_image = rec_image.convert("RGB").resize(model_input_size)

        return np.asarray(rec_image).astype(np.float32)

    def step(self, action):
        done_flag = False

        # info = {'curr_image': self.curr_image.filename}
        info = {}

        q_map = action.reshape(self.q_block_shape)
        q_table = 1. / q_map

        gt_input = self.curr_image.convert("RGB").resize(model_input_size)
        gt_input = preprocess_input(np.expand_dims(np.asarray(gt_input).astype(np.float32), axis=0))
        gt_result = self.deep_model.predict(gt_input)

        curr_input = preprocess_input(np.expand_dims(self._process_single_image(q_table), axis=0))

        predict_loss, predict_acc = self.deep_model.evaluate(curr_input, gt_result, verbose=2)

        if predict_acc == 0:  # Wrong prediction
            self.curr_reward += 20 * self.endurance + 0.5 ** predict_loss
            done_flag = True
        else:
            self.endurance += 1

        prediction_reward = 0.5 ** predict_loss
        # compression_reward = -np.log(np.sum(action))
        # compression_reward = 0.5 ** np.sum(action)
        compression_reward = 0
        self.curr_reward += 20 * self.endurance + prediction_reward + compression_reward

        info['prediction_reward'] = prediction_reward
        info['compression_reward'] = compression_reward

        self.curr_image = next(self.generator)
        features = self.get_features()

        if self.endurance >= len(self.training_image_file_names):  # Finished all images
            done_flag = True

        info['predict_loss'] = predict_loss
        info['action_sum'] = np.sum(action)
        info['endurance'] = self.endurance
        info['reward'] = self.curr_reward

        return features, self.curr_reward, done_flag, info
