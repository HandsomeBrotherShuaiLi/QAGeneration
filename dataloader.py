'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: dataloader.py
@time: 12/9/20 17:37
@desc:
'''
import json, tqdm
import numpy as np
import codecs


class Dataloader(object):
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
        data = json.load(open('data/train.json', 'r', encoding='utf-8'))
        for k in tqdm.tqdm(data.keys()):
            sample = data[k]
            text = sample['text']
            answer_index = sample['answer_index']
            question_index = sample['question_index']
            ids, segments = self.tokenizer.encode(text, max_len=512)
            all_samples.append([ids, segments, answer_index, question_index])
        return all_samples

    def generator(self,is_train=True):
        data = self.train_data if is_train else self.val_data
        index = np.array(range(len(data)))
        np.random.shuffle(index)
        start = 0
        while True:
            batch_ids = np.zeros(shape=(self.batch_size,512))
            batch_segs = np.zeros(shape=(self.batch_size, 512))
            batch_answer_index = np.zeros(shape=(self.batch_size, 512))
            batch_question_index = np.zeros(shape=(self.batch_size, 512))
            if start + self.batch_size < len(index):
                batch_index = index[start:(start + self.batch_size)]
            else:
                batch_index = np.hstack((index[start:],
                                         index[:(self.batch_size - len(index[start:]))]))
            for i,j in enumerate(batch_index):
                sample = data[j]
                batch_ids[i,:] = sample[0]
                batch_segs[i,:] = sample[1]
                batch_answer_index[i, sample[2][0]:sample[2][1]] = 1
                batch_question_index[i, sample[3][0]:sample[3][1]] = 1
            batch_question_index = np.expand_dims(batch_question_index, axis=-1)
            batch_answer_index = np.expand_dims(batch_answer_index, axis=-1)
            yield [batch_ids, batch_segs, batch_answer_index], batch_question_index
            start = (start + self.batch_size) % len(index)
