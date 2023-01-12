#!/usr/bin/env python
import keras
import tensorflow as tf
tf.random.set_seed(42)
import sklearn, wandb
import config as cfg
import numpy as np
from sklearn.linear_model import LinearRegression
cfg.set_seed(3407)

X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([0, -1, -2, -3], dtype=np.float32)

wandb.init(

    job_type='Experiment1',
    project='Gradient Descent',
    # dir='/root/autotmp-dl/All_About_Question_Answering/wandb/',
    entity='unicorn_upc_dl',
    config=cfg.wandb_config,
    # sync_tensorboard=True,
    name='learning rate ' + str(cfg.wandb_config['learning_rate'])
         + ' ' + 'optimizer ' + 'GradientDescentOptimizer',
    # notes='',
    ####
)

config = wandb.config
learning_rate = config.learning_rate


def linear_regression(X, y):
    X = X.reshape(-1, 1)

    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    coefficient = reg.coef_
    intercept = reg.intercept_

    wandb.log({
        'regression_score': score,
        'regression_coefficient ': reg.coef_,
        'regression_intercept': reg.intercept_,
    })
    return score, coefficient, intercept


class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, init_weight=0.3, **kwargs):
        super(GradientLayer, self).__init__()
        self.units = 1
        self.init_weight = init_weight

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=[input_shape[-1], self.units],
            # initializer=tf.Variable(self.init_weight,dtype=tf.float32),
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=[self.units],
            # initializer = tf.Variable(0.0,dtype=tf.float32),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        print(self.weight, self.bias, inputs, (inputs @ self.weight + self.bias), '\n')
        return inputs @ self.weight + self.bias


model = keras.Sequential(
    [GradientLayer(1, input_shape=[1])]
)

x_train = tf.expand_dims(tf.convert_to_tensor([1, 2, 3, 4], dtype=tf.float32), axis=1)
y_train = tf.expand_dims(tf.convert_to_tensor([0, -1, -2, -3], dtype=tf.float32), axis=1)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate, use_locking=False, name='GradientDescent'
)
# optimizer = Adam(learning_rate=learning_rate)
# optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)


model.compile(loss="mse", optimizer=optimizer, metrics='mse')

history = model.fit(x_train, y_train, epochs=config.epochs)


print(model.trainable_variables)
print(model.get_weights())
model.summary()

score, coefficient, intercept = linear_regression(X, y)

print(
    f'regression_score: {score}',
    f'regression_coefficient: {coefficient}',
    f'regression_intercept: {intercept}',
)


