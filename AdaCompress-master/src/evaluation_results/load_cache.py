from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
import time

if __name__ == '__main__':
    with open("imagenet_face_plusplus_ref2000.pkl","rb") as file:
        ref = pk.load(file)
    ref1 = list(ref.items())
    ref1_num = len(ref1)
    with open("image_reference_cache_face.defaultdict","rb") as file:
        reg = pk.load(file)
    reg1 = list(reg.items())
    reg1_num = len(reg1)
    image_path = [i[0] for i in list(ref.items())]
    result = [i[1] for i in list(ref.items())]
    reg_image_path = [i[0] for i in list(reg.items())]
    reg_result = [i[1] for i in list(reg.items())]
    print(image_path[0:5])
    print(reg_image_path[0:5])
    print(result[0:5])
    print(reg_result[0:5])

    with open("image_reference_cache.defaultdict","rb") as file:
        image_cache = pk.load(file)
    image_cache1 = list(image_cache.items())
    image_cache1_num = len(image_cache1)

    with open("face_train_log.defaultdict","rb") as file:
        train_log = pk.load(file)
    train_path = train_log["image_path"]
    train_acc_r = train_log["acc_r"]
    train_size_r = train_log["size_r"]
    train_epsilon = train_log["epsilon"]
    print("train_num ",len(train_path))
    print("acc ",np.mean(train_acc_r))
    print("size ",np.mean(train_size_r))
    print("epsilon ",np.mean(train_epsilon))
    print("epsilon100 ",np.mean(train_epsilon[:100]))
    # np.mean(train_acc_r[:200])