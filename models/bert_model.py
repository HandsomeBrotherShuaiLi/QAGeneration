'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: models.py
@time: 12/9/20 17:37
@desc:
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)

import keras, tqdm
import codecs, json
import numpy as np
from keras_bert import (load_trained_model_from_checkpoint,
                        get_pretrained, PretrainedList,
                        get_checkpoint_paths, Tokenizer)
from dataloader import Dataloader


def download_pretrained_bert(language_backbone='chinese_wwm_base'):
    base_model_path = {
        'multi_cased_base': PretrainedList.multi_cased_base,
        'chinese_base': PretrainedList.chinese_base,
        'wwm_uncased_large': PretrainedList.wwm_uncased_large,
        'wwm_cased_large': PretrainedList.wwm_cased_large,
        'chinese_wwm_base': 'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip',
        'bert_base_cased': 'https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip',
        'bert_large_cased': 'https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip',
        'bert_base_uncased': 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip',
        'bert_large_uncased': 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip'
    }
    model_path = get_pretrained(base_model_path[language_backbone.lower()])
    paths = get_checkpoint_paths(model_path)
    print(paths)
    return paths


class QAGenerator(object):
    def __init__(self,
                 with_bert=True,
                 fine_tune=True,
                 language_backbone='chinese_wwm_base',
                 batch_size=16):
        """

        :param with_bert:
        :param fine_tune:
        :param model_arch: pointer, seq2seq or transformer
        """
        self.language_backbone = language_backbone
        self.with_bert = with_bert
        self.fine_tune = fine_tune
        self.paths = download_pretrained_bert(language_backbone)
        self.bs = batch_size
        token_dict = {}
        with codecs.open(self.paths.vocab, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = Tokenizer(token_dict)
        self.dataloader = Dataloader(tokenizer=self.tokenizer,
                                     batch_size=self.bs, split_rate=0.1)

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(
            config_file=self.paths.config,
            checkpoint_file=self.paths.checkpoint,
            trainable=self.fine_tune, training=self.fine_tune,
            seq_len=512,
        )
        text_token_input, text_segment_input = bert_model.inputs[:2]
        index_input = keras.layers.Input(shape=(512, 1), name='index_input')
        bert_output = bert_model.get_layer('Encoder-12-FeedForward-Norm').output
        w = keras.layers.Dense(1)(bert_output)
        y = keras.layers.Add()([w, index_input])
        y = keras.layers.Activation(activation='sigmoid')(y)
        model = keras.models.Model([text_token_input, text_segment_input, index_input], y)
        model.summary()
        try:
            keras.utils.plot_model(model)
        except:
            pass
        return model

    def train(self):
        model_name = '_'.join([
            'with_bert', 'fine_tune',
            self.language_backbone
        ])
        model = self.build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss=keras.losses.binary_crossentropy,
            metrics=['acc'],
        )
        model.fit_generator(
            generator=self.dataloader.generator(is_train=True),
            steps_per_epoch=self.dataloader.train_steps,
            validation_data=self.dataloader.generator(False),
            validation_steps=self.dataloader.val_steps,
            verbose=1, shuffle=True, epochs=300, initial_epoch=0,
            callbacks=[
                keras.callbacks.TensorBoard('logs'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=28, verbose=1),
                keras.callbacks.ModelCheckpoint(monitor='val_loss', verbose=1, period=1,
                                                save_weights_only=False, save_best_only=True,
                                                filepath='saved_models/{}'.format(model_name) + '.h5'
                                                )
            ]
        )

    def inference(self):
        model = self.build_model()
        model.load_weights('saved_models/w_i_t_h___b_e_r_t___f_i_n_e___t_u_n_e___c_h_i_n_e_s_e___w_w_m___b_a_s_e.h5')
        data = json.load(open('../data/test.json', 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            sample = data[k]
            text = sample['text']
            answer_index = sample['answer_index']
            ids, segs = self.tokenizer.encode(text, max_len=512)
            ids = np.expand_dims(ids, axis=0)
            segs = np.expand_dims(segs, axis=0)
            idxs = np.zeros(shape=(1, 512))
            idxs[0, answer_index[0]:answer_index[1]] = 1
            idxs = np.expand_dims(idxs, axis=-1)
            pred = np.squeeze(model.predict([ids, segs, idxs])[0], axis=-1)
            pred_idx = np.where(pred >= 0.4)[0].tolist()
            selected_text = []
            print(pred_idx)
            for idx in pred_idx:
                selected_text.append(text[idx])
            print(''.join(selected_text).encode('utf-8'))
            print('*'*100)


if __name__ == '__main__':
    app = QAGenerator()
    app.inference()
