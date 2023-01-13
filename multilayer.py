#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot
from functools import partial
import numpy as np
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam, Adamax, RMSprop
# import tensorflow_datasets as tfds
import wandb
import config as cfg
import keras
from wandb.keras import WandbCallback
import tensorflow_addons as tfa
# from tensorflow.keras.utils import multi_gpu_model
# def plot_history(metrics_lst):
#     for metric in metrics_lst:

tf.debugging.set_log_device_placement(True)
mirrored_strategy = tf.distribute.MirroredStrategy()
gpu_len = len(tf.config.experimental.list_physical_devices('GPU'))
print('gpu_len: ', gpu_len)

tf.random.set_seed(42)  # extra code – ensures reproducibility

train, test = tf.keras.datasets.mnist.load_data()
input_shape = (28, 28, 1)
x_train, y_train = train[0], train[1]
x_train= x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test, y_test = test[0], test[1]
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

wandb.init(
    job_type='Experiment3',
    project='Multiple GPUs',
    #     # dir='/root/autotmp-dl/All_About_Question_Answering/wandb/',
    entity='unicorn_upc_dl',
    config=cfg.wandb_config,
    # sync_tensorboard=True,
    name='1GPU learning rate ' + str(cfg.wandb_config['learning_rate'])
         + ' ' + 'AdamW ' + 'batch_size 128',
    # notes='',
    ####
)

wandb_callback = WandbCallback(
    # save_model=True,
    # save_weights_only=True,
    log_weights=True,
    # log_gradients=True,
)

config = wandb.config
learning_rate = config.learning_rate
weight_decay = config.weight_decay

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

with mirrored_strategy.scope():
    model2 = tf.keras.Sequential(([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.MaxPooling2D(),

            # layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(10),
        ]))

    # optimizer = Adam(learning_rate=learning_rate)
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model2.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

history = model2.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    batch_size=128,
    callbacks=[wandb_callback],
)




