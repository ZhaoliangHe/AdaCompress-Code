from PIL import Image
import numpy as np
import os
from io import BytesIO
import cv2
from keras.utils.np_utils import to_categorical
import keras.backend as K
from collections import defaultdict
from functools import partial
from imagenet.imagenet import imagenet_label2class
from keras.applications import *
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.mobilenetv2 import preprocess_input as mobile_preprocess
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
np.random.seed(2)

def multimodel_preprocess(model_name, input_data):
    if model_name in ['vgg16', 'vgg19', 'resnet50']:
        return preprocess_input(input_data)
    elif model_name in ['inception_v3', 'xception']:
        return preprocess_input(input_data, mode='tf')
    elif model_name in ['mobilenetv2_1.00_224']:
        return mobile_preprocess(input_data)

def JPEG_compress(img, quality):
    f = BytesIO()
    img.save(f, format='JPEG', quality=quality)
    bytes_content = f.getvalue()
    img_size = len(bytes_content)
    
    img_read = Image.open(f)
    return img_read, img_size

def gen_sample_set(father_path, samples_per_class):
    image_paths = []
    image_labels = []
    
    img_classes = os.listdir(father_path)
    for img_class in img_classes:
        for image_name in np.random.choice(os.listdir("%s/%s" % (father_path, img_class)), size=samples_per_class):
            sample_image_path = ("%s/%s/%s" % (father_path, img_class, image_name))
            image_label = imagenet_label2class[image_name.split('_')[0]]
            
            image_paths.append(sample_image_path)
            image_labels.append(image_label)
    return image_paths, image_labels


result_dict = defaultdict(dict)
image_paths, image_labels = gen_sample_set('/home/hsli/imagenet-data/train', 20)

for model in [VGG16(), VGG19(), ResNet50(), Xception(), MobileNetV2(), InceptionV3()]:
# for model in [MobileNetV2()]:
    print("\nTesting %s" % model.name)
    qualities = []
    accuracies = []
    sizes = []

    print("image quality ", end='')
    for quality in range(10, 100, 5):
        print(quality, end=' ')
        
        img_train_datas = []
        img_train_labels = []
        img_sizes = []

        for i in range(len(image_paths)):
            compressed_img, img_size = JPEG_compress(Image.open(image_paths[i]).convert("RGB"), int(quality))
            img_train_datas.append(np.asarray(compressed_img.resize((224, 224))))
            img_sizes.append(img_size)

        img_datas = multimodel_preprocess(model.name, np.array(img_train_datas).astype(np.float32))
        img_labels = to_categorical(image_labels, 1000)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        loss, accuracy = model.evaluate(img_datas, img_labels, batch_size=64, verbose=2)

        qualities.append(quality)
        accuracies.append(accuracy)
        sizes.append(np.mean(img_sizes))
    
    # plot(qualities, accuracies, label=model.name)
    
    result_dict[model.name]['qualities'] = qualities
    result_dict[model.name]['accuracies'] = accuracies
    result_dict[model.name]['aver_sizes'] = sizes
    
# legend()        
with open("JPEG_compression_performance.dict", 'wb') as f:
	pickle.dump(result_dict, f)