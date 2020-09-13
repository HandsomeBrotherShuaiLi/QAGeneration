'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: dataloader.py
@time: 12/9/20 17:37
@desc:
'''
import os
import json, tqdm
import numpy as np
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.utils import to_categorical


class Dataloader_v1(object):
    def __init__(self,
                 tokenizer,
                 batch_size=16,
                 split_rate=0.1):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.rate = split_rate
        self.all_samples = self.process()
        np.random.shuffle(self.all_samples)
        val_num = int(self.rate * len(self.all_samples))
        self.val_data, self.train_data = self.all_samples[:val_num], self.all_samples[val_num:]
        self.val_steps, self.train_steps = len(self.val_data) // batch_size, len(self.train_data) // batch_size
        print('{}->{}\n{}->{}'.format(
            len(self.train_data), self.train_steps, len(self.val_data), self.val_steps)
        )

    def process(self):
        all_samples = []
        data = json.load(open('../../data/train.json', 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            sample = data[k]
            text = sample['text']
            answer_index = sample['answer_index']
            question_index = sample['question_index']
            ids, segments = self.tokenizer.encode(text, max_len=512)
            all_samples.append([ids, segments, answer_index, question_index])
        return all_samples

    def generator(self, is_train=True):
        data = self.train_data if is_train else self.val_data
        index = np.array(range(len(data)))
        np.random.shuffle(index)
        start = 0
        while True:
            batch_ids = np.zeros(shape=(self.batch_size, 512))
            batch_segs = np.zeros(shape=(self.batch_size, 512))
            batch_answer_index = np.zeros(shape=(self.batch_size, 512))
            batch_question_index = np.zeros(shape=(self.batch_size, 512))
            if start + self.batch_size < len(index):
                batch_index = index[start:(start + self.batch_size)]
            else:
                batch_index = np.hstack((index[start:],
                                         index[:(self.batch_size - len(index[start:]))]))
            for i, j in enumerate(batch_index):
                sample = data[j]
                batch_ids[i, :] = sample[0]
                batch_segs[i, :] = sample[1]
                batch_answer_index[i, sample[2][0]:sample[2][1]] = 1
                batch_question_index[i, sample[3][0]:sample[3][1]] = 1
            batch_question_index = np.expand_dims(batch_question_index, axis=-1)
            batch_answer_index = np.expand_dims(batch_answer_index, axis=-1)
            yield [batch_ids, batch_segs, batch_answer_index], batch_question_index
            start = (start + self.batch_size) % len(index)


class Dataloader_v2(object):
    def __init__(self, batch_size, split_rate=0.1,
                 train_data='data/train.json',
                 test_data='data/test.json'):
        self.batch_size = batch_size
        self.split_rate = split_rate
        self.train_data = train_data
        self.test_data = test_data
        if not os.path.exists('data/all_train_samples.pkl') or not os.path.exists(
                'data/all_test_samples.pkl') or not os.path.exists('data/word2idx.json') or not os.path.exists(
            'data/idx2word.json'):
            self.word2idx, self.idx2word, self.all_train_samples, self.all_test_samples = self.process()
            json.dump(self.word2idx, open('data/word2idx.json', 'w', encoding='utf-8'), ensure_ascii=False)
            json.dump(self.idx2word, open('data/idx2word.json', 'w', encoding='utf-8'), ensure_ascii=False)
            pickle.dump(self.all_train_samples, open('data/all_train_samples.pkl', 'wb'))
            pickle.dump(self.all_test_samples, open('data/all_test_samples.pkl', 'wb'))
        else:
            self.word2idx = json.load(open('data/word2idx.json', 'r', encoding='utf-8'))
            self.idx2word = json.load(open('data/idx2word.json', 'r', encoding='utf-8'))
            self.all_train_samples = pickle.load(open('data/all_train_samples.pkl', 'rb'))
            self.all_test_samples = pickle.load(open('data/all_test_samples.pkl', 'rb'))
        val_num = int(self.split_rate * len(self.all_train_samples))
        np.random.shuffle(self.all_train_samples)
        self.train_samples = self.all_train_samples[val_num:]
        self.val_samples = self.all_train_samples[:val_num]
        self.train_steps = len(self.train_samples) // self.batch_size
        self.val_steps = len(self.val_samples) // self.batch_size
        print('{} train data, {} val data, {} train steps, {} val steps'.format(
            len(self.train_samples), len(self.val_samples), self.train_steps, self.val_steps
        ))

    def process(self):
        all_train_samples = []
        all_test_samples = []
        char_dict = defaultdict(int)
        word2idx = {'pad': 0}
        idx2word = {0: 'pad'}
        data = json.load(open(self.train_data, 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            text = list(data[k]['text'])
            question = list(data[k]['question'])
            answer = list(data[k]['answer'])
            for i in text + question + answer:
                char_dict[i] += 1
        data = json.load(open(self.test_data, 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            text = list(data[k]['text'])
            answer = list(data[k]['answer'])
            for i in text + answer:
                char_dict[i] += 1
        # 0 is for padding
        chars = list(char_dict.keys()) + ['answer_beg', 'answer_end', 'question_beg', 'question_end']
        for idx, i in enumerate(chars):
            word2idx[i] = idx + 1
            idx2word[idx + 1] = i
        data = json.load(open(self.train_data, 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            text = data[k]['text']
            answer_index = data[k]['answer_index']
            seg_1, answer, seg_2 = text[:answer_index[0]], text[answer_index[0]:answer_index[1]], text[answer_index[1]:]
            text_ids = []
            for i in seg_1:
                text_ids.append(word2idx[i])
            text_ids.append(word2idx['answer_beg'])
            for i in answer:
                text_ids.append(word2idx[i])
            text_ids.append(word2idx['answer_end'])
            for i in seg_2:
                text_ids.append(word2idx[i])

            question = data[k]['question']
            question_ids = [word2idx['question_beg']]
            for i in question:
                question_ids.append(word2idx[i])
            question_ids.append(word2idx['question_end'])
            question_index = data[k]['question_index']
            new_question_index = None
            if question_index[1] <= answer_index[0]:
                new_question_index = question_index
            elif question_index[0] >= answer_index[1]:
                new_question_index = [question_index[0] + 2, question_index[1] + 2]
            elif question_index[0] < answer_index[0] and question_index[1] <= answer_index[1]:
                new_question_index = [question_index[0], question_index[1] + 1]
            elif question_index[0] < answer_index[0] and question_index[1] > answer_index[1]:
                new_question_index = [question_index[0], question_index[1] + 2]
            elif question_index[0] >= answer_index[0] and question_index[1] <= answer_index[1]:
                new_question_index = [question_index[0] + 1, question_index[1] + 1]
            elif question_index[0] >= answer_index[0] and question_index[1] > answer_index[1]:
                new_question_index = [question_index[0] + 1, question_index[1] + 2]

            all_train_samples.append(
                [text, question, answer, answer_index, question_index,
                 new_question_index, text_ids, question_ids
                 ]
            )

        data = json.load(open(self.test_data, 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            text = data[k]['text']
            answer_index = data[k]['answer_index']
            seg_1, answer, seg_2 = text[:answer_index[0]], text[answer_index[0]:answer_index[1]], text[answer_index[1]:]
            text_ids = []
            for i in seg_1:
                text_ids.append(word2idx[i])
            text_ids.append(word2idx['answer_beg'])
            for i in answer:
                text_ids.append(word2idx[i])
            text_ids.append(word2idx['answer_end'])
            for i in seg_2:
                text_ids.append(word2idx[i])
            all_test_samples.append(
                [
                    text, answer, answer_index, text_ids
                ]
            )

        return word2idx, idx2word, all_train_samples, all_test_samples

    def generator(self, is_train=True, is_custom=False):
        data = self.train_samples if is_train else self.val_samples
        index = np.array(range(len(data)))
        np.random.shuffle(index)
        start = 0
        if not is_custom:
            while True:
                batch_inputs_for_text = []
                batch_outputs = []
                if start + self.batch_size < len(index):
                    batch_index = index[start:(start + self.batch_size)]
                else:
                    batch_index = np.hstack((index[start:],
                                             index[:(self.batch_size - len(index[start:]))]))
                for i, j in enumerate(batch_index):
                    sample = data[j]
                    text_ids, question_ids = sample[-2], sample[-1]
                    batch_inputs_for_text.append(text_ids)
                    batch_outputs.append(question_ids)
                batch_inputs_for_text = np.array(pad_sequences(batch_inputs_for_text,
                                                               padding='post', truncating='post'))
                batch_outputs = pad_sequences(batch_outputs, maxlen=batch_inputs_for_text.shape[1],
                                              padding='post', truncating='post')
                batch_outputs = to_categorical(batch_outputs,
                                               num_classes=len(self.word2idx.keys()))
                yield batch_inputs_for_text, batch_outputs
                start = (start + self.batch_size) % len(index)
        else:
            while True:
                batch_encoder_inputs = []
                batch_target_outputs = []
                batch_question = []
                batch_decoder_inputs = []
                if start + self.batch_size < len(index):
                    batch_index = index[start:(start + self.batch_size)]
                else:
                    batch_index = np.hstack((index[start:],
                                             index[:(self.batch_size - len(index[start:]))]))
                for i, j in enumerate(batch_index):
                    sample = data[j]
                    new_question_index, text_ids, question_ids = sample[-3], sample[-2], sample[-1]
                    question_code = [0] * len(text_ids)
                    for z in range(new_question_index[0], new_question_index[1]):
                        question_code[z] = 1
                    batch_question.append(question_code)
                    batch_encoder_inputs.append(text_ids)
                    batch_decoder_inputs.append(question_ids)
                    batch_target_outputs.append(question_ids[1:] + [0])
                batch_encoder_inputs = np.array(pad_sequences(batch_encoder_inputs,
                                                              padding='post', truncating='post'))

                batch_decoder_inputs = np.array(pad_sequences(batch_decoder_inputs,
                                                              padding='post', truncating='post'))

                batch_question = np.array(pad_sequences(batch_question,
                                                        padding='post', truncating='post'))
                batch_question = np.expand_dims(batch_question, axis=-1)

                batch_target_outputs = pad_sequences(batch_target_outputs,
                                                     padding='post', truncating='post')
                batch_target_outputs = to_categorical(batch_target_outputs,
                                                      num_classes=len(self.word2idx.keys()))

                yield [batch_encoder_inputs, batch_decoder_inputs], [batch_target_outputs, batch_question]
                start = (start + self.batch_size) % len(index)
