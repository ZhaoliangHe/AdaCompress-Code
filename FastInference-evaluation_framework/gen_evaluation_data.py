from res_manager import ResultManager

from keras.applications import ResNet50
import numpy as np
import time
from keras.utils.np_utils import to_categorical
from imagenet import imagenet_label2class
from keras.applications.resnet50 import preprocess_input
import os
from io import BytesIO
from collections import defaultdict
from PIL import Image

def gen_sample_set(imagenet_data_path, samples_per_class):
    image_paths = []
    image_labels = []

    img_classes = os.listdir(imagenet_data_path)
    for img_class in img_classes:
        for image_name in np.random.choice(os.listdir("%s/%s" % (imagenet_data_path, img_class)),
                                           size=samples_per_class):
            sample_image_path = ("%s/%s/%s" % (imagenet_data_path, img_class, image_name))
            image_label = imagenet_label2class[image_name.split('_')[0]]

            image_paths.append(sample_image_path)
            image_labels.append(image_label)
    return image_paths, image_labels

def compress_Q(img, Q):
    f = BytesIO()
    img.save(f, format="JPEG", quality=Q)
    size = len(f.getvalue())
    return Image.open(f), size

def max_continous_idx(l):
    sums = []
    ids = []
    curr_sum = 0
    for idx, item in enumerate(l):
        if item == 1:
            curr_sum += 1
            if curr_sum == len(l):
                return len(l) - 1
        else:
            sums.append(curr_sum)
            curr_sum = 0
            ids.append(idx -  1)
    return np.array(ids)[np.argmax(sums)]

rm = ResultManager('evaluation_results')

sample_paths, sample_labels = gen_sample_set('/home/hsli/gnode02/imagenet-data/train/', 5)
model_labels = to_categorical(sample_labels, 1000)

model = ResNet50()
model.compile('adam', 'categorical_crossentropy', ['top_k_categorical_accuracy'])

robust_dict = defaultdict(list)
measurement_dict = defaultdict(list)
banchmark = defaultdict(list)
for idx, path in enumerate(sample_paths):
    if idx % 100 == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end='\t')
        print(idx)
    model_label = np.expand_dims(model_labels[idx], axis=0)
    top5_list = []
    size_list = []
    img = Image.open(path).convert("RGB")
    for i in np.arange(100, 0, -1):
        compressed_img, size = compress_Q(img, int(i))
        model_input_data = preprocess_input(
            np.expand_dims(np.asarray(compressed_img.resize((224, 224)), dtype=np.float32), axis=0))
        loss, top5 = model.evaluate(model_input_data, model_label, verbose=0)
        top5_list.append(int(top5))
        size_list.append(size)
    minimal_stable_q = 100 - max_continous_idx(top5_list)
    measurement_dict['paths'].append(path)
    measurement_dict['top5_upon_q'].append(top5_list)
    measurement_dict['minimal_stable_q'].append(minimal_stable_q)
    measurement_dict['size_lists'].append(size_list)

    if minimal_stable_q < 100:
        robust_dict['paths'].append(path)
        robust_dict['robustness'].append(minimal_stable_q)

top5_data = np.array(measurement_dict['top5_upon_q'])
top5_upon_q_list = []
for i in range(100):
    top5_upon_q_list.append(sum(top5_data[:, i]) / len(top5_data[:, i]))

banchmark['accuracies'] = top5_upon_q_list[::-1]
banchmark['qualities'] = [i+1 for i in range(100)]
banchmark['sizes'] = np.mean(np.array(measurement_dict['size_lists']), axis=0)[::-1].tolist()

rm.save(banchmark, name="Banchmark_JPEG_ResNet50", topic="Banchmark", comment="accuracies and sizes upon each Q, JPEG and ResNet50 on 5K")
rm.save(robust_dict, name="Robustness_JPEG_ResNet50", topic="Dataset", comment="robust dataset for JPEG and ResNet50(top-5)")