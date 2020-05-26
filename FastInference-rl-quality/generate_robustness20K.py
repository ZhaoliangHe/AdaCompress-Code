from res_manager import ResultManager
import numpy as np
from PIL import Image
import time
import os
import pickle
from collections import defaultdict
from io import BytesIO

from keras.applications import InceptionV3
from keras.utils.np_utils import to_categorical
from imagenet import imagenet_label2class
from keras.applications.inception_v3 import preprocess_input

rm = ResultManager('results')

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
    return Image.open(f)

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
	
	
sample_paths, sample_labels = gen_sample_set('/home/hsli/gnode02/imagenet-data/train/', 22)

model_labels = to_categorical(sample_labels, 1000)

model = InceptionV3()

model.compile('adam', 'categorical_crossentropy', ['accuracy', 'top_k_categorical_accuracy'])

rm.print_meta_info()

robust_dict = defaultdict(list)
measurement_dict = defaultdict(list)
for idx, path in enumerate(sample_paths):
    if idx % 100 == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end='\t')
        print(idx)
    model_label = np.expand_dims(model_labels[idx], axis=0)
    top5_list = []
    top1_list = []
    img = Image.open(path).convert("RGB")
    for i in np.arange(100, 0, -1):
        model_input_data = preprocess_input(np.expand_dims(np.asarray(compress_Q(img, int(i)).resize((299, 299)), dtype=np.float32), axis=0))
        loss, top1, top5 = model.evaluate(model_input_data, model_label, verbose=0)
        top5_list.append(int(top5))
        top1_list.append(int(top1))
    minimal_stable_q = 100 - max_continous_idx(top5_list)
    measurement_dict['paths'].append(path)
    measurement_dict['top1_upon_q'].append(top1_list)
    measurement_dict['top5_upon_q'].append(top5_list)
    measurement_dict['minimal_stable_q'].append(minimal_stable_q)
    
    if minimal_stable_q < 100:
        robust_dict['paths'].append(path)
        robust_dict['robustness'].append(minimal_stable_q)
    
rm.save(measurement_dict, name="measurement imagenet", topic="measurements", comment="top1 and top5 acc and upon different quality in random 22k imgs on inceptionv3")
rm.save(robust_dict, name="robust dict 22k", topic="measurements", comment="robustness in random 22k imgs upon different q on inceptionv3")
