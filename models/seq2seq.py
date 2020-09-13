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
    def __init__(self, vocab_num, model='final',
                 hidden_size=64):
        self.vocab_num = vocab_num
        self.hidden = hidden_size
        self.model = model

    def build_model(self):
        encoder_input = keras.layers.Input(shape=(None,), name='encoder_input')
        decoder_input = keras.layers.Input(shape=(None,), name='decoder_input')

        embedding_layer = keras.layers.Embedding(self.vocab_num, self.hidden)
        encoder_embed = embedding_layer(encoder_input)
        decoder_embed = embedding_layer(decoder_input)

        encoder_pos_embed = PositionEmbedding(name='encoder_position_embedding')(encoder_embed)
        encoder_embed = keras.layers.Concatenate()([encoder_embed, encoder_pos_embed])

        decoder_pos_embed = PositionEmbedding(name='decoder_position_embedding')(decoder_embed)
        decoder_embed = keras.layers.Concatenate()([decoder_embed, decoder_pos_embed])

        encoder, state_h, state_c = keras.layers.LSTM(self.hidden * 2, return_state=True, return_sequences=True)(
            encoder_embed)
        encoder_states = [state_h, state_c]

        decoder, _, _ = keras.layers.LSTM(self.hidden * 2, return_sequences=True, return_state=True)(
            decoder_embed, initial_state=encoder_states)

        question_prob = keras.layers.Dense(1, activation='sigmoid',
                                           name='question_location')(encoder)
        decoder_prediction = keras.layers.Dense(self.vocab_num, activation='softmax',
                                                name='decoder')(decoder)
        model = keras.Model([encoder_input, decoder_input], [decoder_prediction, question_prob])
        model.summary()
        try:
            keras.utils.plot_model(model, './final_model.png', show_layer_names=True,
                                   show_shapes=True)
        except:
            pass
        return model


def train(model='final', batch_size=32):
    dataloader = Dataloader_v2(batch_size=batch_size)
    app = QAGenerator(model=model,
                      vocab_num=len(dataloader.word2idx.keys()))
    seq2seq_model = app.build_model()
    seq2seq_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={'decoder': keras.losses.categorical_crossentropy,
              'question_location': keras.losses.binary_crossentropy},
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


if __name__ == '__main__':
    app = QAGenerator(vocab_num=4152)
    app.build_model()