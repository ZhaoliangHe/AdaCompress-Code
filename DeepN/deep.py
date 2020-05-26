from numpy import *
import numpy as np
from PIL import Image
import os
from io import BytesIO
from keras.preprocessing import image
import cv2
import hashlib
import time
import matplotlib.pyplot as plt
import base64

def split_88(image_data):
    blocks = []
    #image_size = image_data.shape[0]
    image_size = min(image_data.shape[0],image_data.shape[1])
    for i in range(int(image_size / 8)):
        row = image_data[8*i:8*(i+1), :]
        for j in range(int(image_size / 8)):
            col = row[:, 8*j:8*(j+1)]
            blocks.append(col.astype(float32))
    return array(blocks)

def merge_88(blocks):
    block_count = blocks.shape[0]
    edge_len = int(sqrt(block_count))
    img_size = 8*edge_len
    image_rows = []
    for i in range(edge_len):
        img_row = hstack(blocks[edge_len*i: edge_len*(i+1), ...])
        image_rows.append(img_row)
    return vstack(image_rows)

def DeepN(crop_image):
    f = BytesIO()
    crop_image.save(f, format='JPEG', quality=75)
    bytes_content = f.getvalue()
    crop_size = len(bytes_content)
    f.seek(0)
    img1 = Image.open(f)

    # DeepN-JPEG
    img_yuv = img1.convert('YCbCr')
    channels = img_yuv.split()

    rec_channels = []
    q_tables = []
    for channel in channels:
        # split and shift
        blocks = split_88(asarray(channel)) - 128
        # dct and get std_matrix
        dct_blocks = array([cv2.dct(item) for item in blocks])
        std_matrix = zeros([8, 8])
        for i in range(8):
            for j in range(8):
                std_matrix[i, j] = std(dct_blocks[:, i, j], ddof=1)

        # Build Q-table
        q_table = zeros([8, 8], dtype=int64)
        a = 255
        b = 80
        c = 240
        T1 = 20
        T2 = 60
        k1 = 9.75
        k2 = 1
        k3 = 3
        Qmin = 5
        for i in range(8):
            for j in range(8):
                std_value = std_matrix[i, j]
                if std_value <= T1:
                    q = a - k1 * std_value
                elif std_value > T2:
                    q = c - k3 * std_value
                else:
                    q = b - k2 * std_value
                q = (q if (q > Qmin) else Qmin)
                q_table[i, j] = q

        q_tables.append(q_table)

        # quantization by q-table and recovery from q-table
        quantizd_dct_blocks = np.round(array([block / q_table for block in dct_blocks]))
        rec_dct_blocks = array([block * q_table for block in quantizd_dct_blocks])
        # idct, round and shift
        idct_rec_blocks = np.round(array([cv2.idct(block) for block in rec_dct_blocks])) + 128
        # merge into an image
        rec_channel = merge_88(idct_rec_blocks)

        rec_channels.append(clip(rec_channel, 0, 255))

    rec_channels = array(rec_channels, dtype=uint8)

    pil_channels = [Image.fromarray(channel) for channel in rec_channels]

    # image and compress image
    # rec_image = Image.merge("YCbCr", channels).convert('RGB')
    rec_pil_image = Image.merge("YCbCr", pil_channels).convert('RGB')

    # compute file size
    # rec_pil_image
    # f2 = BytesIO()
    # rec_pil_image.save(f2, format='JPEG', quality=75)  # qtables=pil_qtables)
    # rec_pil_image_size = len(f2.getvalue())
    return rec_pil_image

if __name__ == '__main__':
    image = Image.open('/home/imagenet-data/train//n03977966/n03977966_11273.JPEG')
    image_size = min(image.size)
    crop_image = image.crop((0, 0, image_size, image_size))

    f1 = BytesIO()
    crop_image.save(f1, format='JPEG', quality=75)  # qtables=pil_qtables)
    print("f1",len(f1.getvalue()))

    deepn_image = DeepN(crop_image)
    f2 = BytesIO()
    deepn_image.save(f2, format='JPEG', quality=75)  # qtables=pil_qtables)
    print("f2",len(f2.getvalue()))