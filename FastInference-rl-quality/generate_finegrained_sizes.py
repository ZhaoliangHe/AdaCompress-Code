import numpy as np
import os
from io import BytesIO
from PIL import Image
from res_manager import ResultManager


def gen_sample_set(imagenet_data_path, samples_per_class):
    image_paths = []

    img_classes = os.listdir(imagenet_data_path)
    for img_class in img_classes:
        for image_name in np.random.choice(os.listdir("%s/%s" % (imagenet_data_path, img_class)),
                                           size=samples_per_class):
            sample_image_path = ("%s/%s/%s" % (imagenet_data_path, img_class, image_name))

            image_paths.append(sample_image_path)
    return image_paths


def size_Q(img, Q):
    f = BytesIO()
    img.save(f, format="JPEG", quality=Q)
    return len(f.getvalue())


rm = ResultManager('results')
sample_set = gen_sample_set('/home/hsli/imagenet-data/train/', 1)
aver_size_list = []
for i in np.arange(1, 101, 1):
    print("Parsing %s" % i)
    aver_size = [size_Q(Image.open(path).convert("RGB"), int(i)) for path in sample_set]
    aver_size_list.append(aver_size)

rm.save(aver_size_list, name='aver_size', topic='measurements', comment="finegrained average size upon different Q on 2W images")
