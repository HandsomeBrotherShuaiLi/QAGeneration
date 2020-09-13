'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: main.py
@time: 13/9/20 18:22
@desc:
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
KTF.set_session(session)

from models.seq2seq import train

train(model='custom_simple', batch_size=64)
