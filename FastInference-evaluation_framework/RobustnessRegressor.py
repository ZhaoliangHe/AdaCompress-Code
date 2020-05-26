import numpy as np
from PIL import Image
from multiprocessing import Pool
from res_manager import ResultManager
from image_pipelines import split_88
from keras.applications import MobileNetV2
from keras.utils import multi_gpu_model
from keras.models import Model
from keras import regularizers
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
from keras.applications.mobilenetv2 import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import cv2
import os

tf.set_random_seed(2)
np.random.seed(2)

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

rm = ResultManager('results')

train_size = 19000
test_size = 2000

train_img_paths = [path.replace('gnode02/', '') for path in rm.load(6)['paths'][:train_size]]
train_label_data = np.array([(0., 1.) if item > 3 else (1., 0.) for item in rm.load(6)['robustness'][:train_size]])

test_img_paths = [path.replace('gnode02/', '') for path in rm.load(6)['paths'][-test_size:]]
test_label_data = np.array([(0., 1.) if item > 3 else (1., 0.) for item in rm.load(6)['robustness'][-test_size:]])

# Callback definition
checkpointer = ModelCheckpoint(filepath='checkpoints/partial_mobilenetv2_predictor_2W_acc_{val_acc:.4f}_epoch{epoch:02d}.hdf5',
                               monitor='val_acc',
                               save_best_only=True,
                               mode='auto',
                               period=2,
                               verbose=1)

early_stop = EarlyStopping(monitor='val_acc',
                           min_delta=1e-5,
                           patience=30,
                           verbose=1,
                           mode='auto')

reduce_LR = ReduceLROnPlateau(monitor='acc',
                              factor=0.8,
                              patience=5,
                              mode='auto',
                              verbose=1)

if __name__ == '__main__':
    train_features = np.array([preprocess_input(np.asarray(Image.open(path).convert("RGB").resize((224, 224)), dtype=np.float32)) for path in train_img_paths])
    test_features = np.array([preprocess_input(np.asarray(Image.open(path).convert("RGB").resize((224, 224)), dtype=np.float32)) for path in test_img_paths])

    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[81:98]:
        layer.trainable = True

    x = base_model.layers[[layer.name for layer in base_model.layers].index('block_10_project')].output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    model = multi_gpu_model(model, gpus=2)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Micro fine-tuning...")

    model.fit(train_features,
              train_label_data,
              batch_size=256,
              epochs=300,
              callbacks=[early_stop, reduce_LR, checkpointer],
              verbose=1,
              validation_data=(test_features, test_label_data))

    # for idx, layer in enumerate(model.layers[:100]):
    #     layer.trainable = False
    #
    # for idx, layer in enumerate(model.layers[100:]):
    #     layer.trainable = True
    #
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # print("Fine-tuning...")
    #
    # model.fit(train_features,
    #           train_label_data,
    #           batch_size=128,
    #           epochs=500,
    #           callbacks=[early_stop, reduce_LR],
    #           verbose=1,
    #           validation_data=(test_features, test_label_data))

    print([np.argmax(line) for line in model.predict(test_features[:50])])
#