import pickle
import numpy as np
from keras.applications import VGG16
import cv2
import os
from PIL import Image
from jpeg_utils import *

np.set_printoptions(precision=2)

output_size = 224
model = VGG16()


def image_generator(images_dir, target_size=None):
    count = 0
    while True:
        print("Data round %s" % count)
        for filename in os.listdir(images_dir):
            # img = cv2.imread('%s/%s' % (images_dir, filename))
            img = Image.open('%s/%s' % (images_dir, filename)).convert("RGB").resize(target_size)

            yield img
        count += 1


generator = image_generator('/home/hsli/imagenet-data/train/n03085013', target_size=(output_size, output_size))

with open("q_table.data", 'rb') as f:
    q_table = pickle.load(f)


def q_pipeline(q_table, image):
    y, u, v = image.convert("YCbCr").split()
    _u = u.resize((int(output_size / 2), int(output_size / 2))).resize((output_size, output_size))
    _v = v.resize((int(output_size / 2), int(output_size / 2))).resize((output_size, output_size))

    y_blocks = split_88(np.asarray(y)) - 128
    y_dct_blocks = np.array([cv2.dct(block) for block in y_blocks])
    y_div_blocks = np.array([np.round(block / q_table) for block in y_dct_blocks])

    y_scale_blocks = np.array([block * q_table for block in y_div_blocks])
    y_rec_blocks = np.array([cv2.idct(block) for block in y_scale_blocks]) + 128

    y = merge_88(y_rec_blocks)

    image_rec = Image.merge("YCbCr", [Image.fromarray(y.astype(np.uint8)), _u, _v]).convert("RGB")
    return image_rec


while True:
    origin_image = next(generator)
    origin_image_data = np.asarray(origin_image.convert('RGB'))

    pipeline_image = q_pipeline(q_table, origin_image)
    pipeline_image_data = np.asarray(pipeline_image)

    label = np.argmax(model.predict(np.expand_dims(pipeline_image_data, axis=0))[0])

    cv2.imshow('Original', cv2.cvtColor(origin_image_data, cv2.COLOR_RGB2BGR))
    cv2.imshow('Pipeline %s' % label, cv2.cvtColor(pipeline_image_data, cv2.COLOR_RGB2BGR))
    k = cv2.waitKey()
    if k == 27:
        break

cv2.destroyAllWindows()

print(q_table)
