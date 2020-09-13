'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: checker.py
@time: 12/9/20 23:35
@desc:
'''

import json, tqdm


def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p - maxNum:p], p - maxNum, p


train = json.load(open('data/round1_train_0907.json', 'r', encoding='utf-8'))
train_samples = {}
n = 0
for item in tqdm.tqdm(train):
    text = item['text']
    annos = item['annotations']
    for anno in annos:
        question = anno['Q']
        answer = anno['A']
        _, qs, qe = getNumofCommonSubstr(text, question)
        if answer in text:
            res = answer
            a_s = text.find(answer)
            a_e = a_s + len(answer)
        else:
            res, a_s, a_e = getNumofCommonSubstr(text, answer)
        train_samples[n] = {
            'text': text,
            'question': question,
            'answer': answer,
            'question_index': [qs, qe],
            'answer_index': [a_s, a_e]
        }
        n += 1

test = json.load(open('data/round1_test_0907.json', 'r', encoding='utf-8'))
test_samples = {}
n = 0
for item in tqdm.tqdm(test):
    text = item['text']
    annos = item['annotations']
    for anno in annos:
        answer = anno['A']
        if answer in text:
            res = answer
            a_s = text.find(answer)
            a_e = a_s + len(answer)
        else:
            res, a_s, a_e = getNumofCommonSubstr(text, answer)
        test_samples[n] = {
            'text': text,
            'answer': answer,
            'answer_index': [a_s, a_e]
        }
        n += 1

json.dump(train_samples, open('data/train.json', 'w', encoding='utf-8'),
          ensure_ascii=False)
json.dump(test_samples, open('data/test.json', 'w', encoding='utf-8'),
          ensure_ascii=False)
