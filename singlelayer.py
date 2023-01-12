#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam, Adamax, RMSprop
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import config as cfg
import logging
import keras
from wandb.keras import WandbCallback
import tensorflow_addons as tfa

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# building model and train the model at different learning rate
def train_lr():
    histories = []
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    lr_names = " 0.1 0.01 0.001 0.0001"
    for learning_rate in learning_rates:

        wandb.init(
            job_type='Experiment2',
            project='Single Layer NN2',
            # dir='/root/autotmp-dl/All_About_Question_Answering/wandb/',
            entity='unicorn_upc_dl',
            config=cfg.wandb_config,
            # sync_tensorboard=True,
            name='learning rate ' + str(learning_rate),
            # notes='',
            ####
        )
        wandb_callback = WandbCallback(
            save_model=True,
            # save_weights_only=True,
            # log_weights=True,
            # log_gradients=True,
        )

        optimizer = Adam(learning_rate=learning_rate)
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10)
        ])

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        history = model.fit(
            ds_train,
            epochs=10,
            validation_data=ds_test,
            callbacks=[wandb_callback],
        )
        histories.append(history)
    return histories, lr_names


def build_optims():
    opt_names = "SGD Momentum Nesterov AdaGrad RMSProp Adam Adamax Adadelta Nadam AdamW"
    optim_sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
    optim_momentum = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    optim_nesterov = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    optim_adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.001)

    optim_rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    optim_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    optim_adamax = tf.keras.optimizers.Adamax(learning_rate=0.001)
    optim_adadelta = tf.keras.optimizers.Adadelta(learning_rate=0.001)

    optim_nadam = tf.keras.optimizers.Nadam(learning_rate=0.001)
    optim_adamw = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=0.001,
                                     beta_1=0.9, beta_2=0.999)

    optims = [optim_sgd, optim_momentum, optim_nesterov, optim_adagrad, optim_rmsprop, optim_adam,
              optim_adamax, optim_adadelta, optim_nadam, optim_adamw]

    return optims, opt_names


# building model and train the model at different optimizer
# We simply adpot lr = 0.001 for all configuration
def train_optim(optims, opt_names):
    histories = []
    for optimizer, name in zip(optims, opt_names.split()):

        wandb.init(
            job_type='Experiment2',
            project='Single Layer NN',
            # dir='/root/autotmp-dl/All_About_Question_Answering/wandb/',
            entity='unicorn_upc_dl',
            config=cfg.wandb_config,
            # sync_tensorboard=True,
            name='optimizer ' + name,
            # notes='',
            ####
        )
        wandb_callback = WandbCallback(
            save_model=True,
            # save_weights_only=True,
            # log_weights=True,
            # log_gradients=True,
        )

        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10)
        ])

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        history = model.fit(
            ds_train,
            epochs=10,
            validation_data=ds_test,
            callbacks=[wandb_callback],
        )
        histories.append(history)
    return histories, opt_names


def plot_loss(histories, names, losses=('loss', 'val_loss')):
    for item in losses:
        plt.figure(figsize=(12, 8))
        for history, opt_name in zip(histories, names.split()):
            plt.plot(history.history[item], label=f"{opt_name}", linewidth=3)

        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel({
                    "loss": "Training loss", "val_loss": "Validation loss",
                    }[item])
        plt.legend(loc="upper left")
        plt.axis([0, 10, 0, 2])
        plt.show()


def plot_accs(histories, names, accs=('acc', 'val_acc')):
    for item in accs:
        plt.figure(figsize=(12, 8))
        for history, opt_name in zip(histories, names.split()):
            plt.plot(history.history[item], label=f"{opt_name}", linewidth=3)

        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel({
                    "acc": "Training accuracy", "val_acc": "Validation accuracy",
                    }[item])
        plt.legend(loc="upper left")
        plt.axis([0, 10, 0.7, 1])
        plt.show()


def train_plot_lrs():
    histories, lr_names = train_lr()
    # plot_loss(histories, lr_names)
    # plot_accs(histories, lr_names)


def train_plot_optims():
    optims, opt_names = build_optims()
    histories, opt_names = train_optim(optims, opt_names)
    # plot_loss(histories, opt_names)
    # plot_accs(histories, opt_names)


if __name__ == '__main__':
    train_plot_lrs()
    train_plot_optims()



