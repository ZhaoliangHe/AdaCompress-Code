from jpeg_utils import *
import cv2
from PIL import Image

from keras.applications import VGG16
from keras.models import Model


def image_feeding_pipeline(image_paths, output_size):
    """
    Return batches for training from image paths,
    :param image_paths: list, list of image paths
    :param output_size: output image size (width, height)
    :return: RGB images of shape (batch_size, width, height, channel),
             V channel of all image at the shape of (batch_size, width, height),
             U channel of all image at the shape of (batch_size, width, height),
             Y channel batch of shape (batch_size, n_blocks, 8, 8)
    """
    rgb_images, u_list, v_list, y_dct_list = [], [], [], []
    for image_path in image_paths:
        rgb_image = Image.open(image_path).convert("RGB").resize(output_size)
        rgb_images.append(np.asarray(rgb_image))
        y, u, v = rgb_image.convert("YCbCr").split()

        # Down sampling and up sampling u and v channel
        u_data = np.asarray(u.resize((int(output_size[0] / 2), int(output_size[1] / 2))).resize(output_size))
        v_data = np.asarray(v.resize((int(output_size[0] / 2), int(output_size[1] / 2))).resize(output_size))

        u_list.append(u_data)
        v_list.append(v_data)

        # split and DCT on y channel
        y_data = np.asarray(y)
        y_blocks = split_88(y_data) - 128
        y_dct_blocks = np.array([cv2.dct(block) for block in y_blocks])
        y_dct_list.append(y_dct_blocks)

    base_model = VGG16()
    topless_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    gt_values = topless_model.predict(np.array(rgb_images), batch_size=128, verbose=1)

    return np.array(rgb_images), np.array(u_list), np.array(v_list), np.array(y_dct_list), gt_values
