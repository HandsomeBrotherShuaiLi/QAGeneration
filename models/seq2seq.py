'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: seq2seq.py
@time: 13/9/20 15:41
@desc:
'''
import keras
from models.utils.PostionEmbedding import PositionEmbedding
from models.utils.AttentionDecoder import AttentionDecoder
from models.utils.dataloader import Dataloader_v2


class QAGenerator(object):
    def __init__(self, vocab_num, model,
                 hidden_size=64):
        self.vocab_num = vocab_num
        self.hidden = hidden_size
        self.model = model

    def build_model(self):
        input_layer = keras.layers.Input(shape=(None,), dtype='int64')
        embed = keras.layers.Embedding(self.vocab_num, self.hidden)(input_layer)
        pos_embed = PositionEmbedding()(embed)
        embed = keras.layers.Concatenate()([embed, pos_embed])
        encoder, state_h, state_c = keras.layers.LSTM(self.hidden * 2, return_state=True, return_sequences=True)(
            embed)
        encoder_states = [state_h, state_c]

        if 'custom' in self.model:
            prob = keras.layers.Dense(1, activation='sigmoid', name='question_prob')(encoder)  # loss
            concat_encoder = keras.layers.Concatenate(name='add_mask_prob')([prob, encoder])
            if 'attention' in self.model:
                decoder = AttentionDecoder(self.hidden * 2, self.vocab_num,
                                           embedding_dim=self.hidden,
                                           is_monotonic=False,
                                           normalize_energy=False, name='decoder')(concat_encoder)
            else:
                decoder, _, _ = keras.layers.LSTM(self.hidden * 2, return_state=True, return_sequences=True)(
                    concat_encoder,
                    initial_state=encoder_states)
                decoder = keras.layers.Dense(self.vocab_num, activation='softmax', name='decoder')(decoder)
            model = keras.Model(input_layer, [decoder, prob])
            model.summary()
            try:
                keras.utils.plot_model(model, to_file='./seq2seq_{}.png'.format(self.model),
                                       show_shapes=True, show_layer_names=True)
            except:
                pass
            return model
        else:
            if self.model == 'attention':
                decoder = AttentionDecoder(self.hidden * 2, self.vocab_num,
                                           embedding_dim=self.hidden,
                                           is_monotonic=False,
                                           normalize_energy=False)(encoder)
            else:
                decoder, _, _ = keras.layers.LSTM(self.hidden * 2, return_state=True, return_sequences=True)(encoder,
                                                                                                             initial_state=encoder_states)
                decoder = keras.layers.Dense(self.vocab_num, activation='softmax')(decoder)
            model = keras.Model(input_layer, decoder)
            model.summary()
            try:
                keras.utils.plot_model(model, to_file='./seq2seq_{}.png'.format(self.model),
                                       show_shapes=True, show_layer_names=True)
            except:
                pass
            return model


def train(model='simple', batch_size=32):
    dataloader = Dataloader_v2(batch_size=batch_size)
    app = QAGenerator(model=model,
                      vocab_num=len(dataloader.word2idx.keys()))
    seq2seq_model = app.build_model()
    if 'custom' not in model:
        seq2seq_model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss=keras.losses.categorical_crossentropy,
            metrics=['acc'],
        )
        model_name = 'seq2seq_{}'.format(model)
        seq2seq_model.fit_generator(
            generator=dataloader.generator(is_train=True, is_custom=False),
            steps_per_epoch=dataloader.train_steps,
            validation_steps=dataloader.val_steps,
            validation_data=dataloader.generator(is_train=False, is_custom=False),
            epochs=300, initial_epoch=0, verbose=1, shuffle=True,
            callbacks=[
                keras.callbacks.TensorBoard('logs'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=28, verbose=1),
                keras.callbacks.ModelCheckpoint(monitor='val_loss', verbose=1, period=1,
                                                save_weights_only=False, save_best_only=True,
                                                filepath='saved_models/{}.h5'.format(model_name)
                                                )
            ]
        )
    else:
        seq2seq_model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss={'decoder': keras.losses.categorical_crossentropy,
                  'question_prob': keras.losses.binary_crossentropy},
            metrics=['acc'],
        )
        model_name = 'seq2seq_{}'.format(model)
        seq2seq_model.fit_generator(
            generator=dataloader.generator(is_train=True, is_custom=True),
            steps_per_epoch=dataloader.train_steps,
            validation_steps=dataloader.val_steps,
            validation_data=dataloader.generator(is_train=False, is_custom=True),
            epochs=300, initial_epoch=0, verbose=1, shuffle=True,
            callbacks=[
                keras.callbacks.TensorBoard('logs'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=28, verbose=1),
                keras.callbacks.ModelCheckpoint(monitor='val_loss', verbose=1, period=1,
                                                save_weights_only=False, save_best_only=True,
                                                filepath='saved_models/{}.h5'.format(model_name)
                                                )
            ]
        )

