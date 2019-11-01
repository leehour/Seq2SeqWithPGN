import numpy as np
import pandas as pd

from config import train_path, test_path, train_seg_x_path, train_seg_target_path, test_seg_x_path
from utils.split_data import split_data

PIC_TOKEN = '[ 图片 ]'
VOICE_TOKEN = '[ 语音 ]'
LINE_TOKEN = '|'


def read_data(data_path):
    data = pd.read_csv(data_path)
    data_list = np.array(data)
    data_list = data_list.tolist()
    data_x = []
    data_target = []
    for line in data_list:
        if len(line) == 6:
            data_x.append(str(line[3]) + ' ' + str(line[4]))
            data_target.append(line[5])
        elif len(line) == 5:
            data_x.append(str(line[3]) + ' ' + str(line[4]))
    return data_x, data_target


def read_stopwords(stopword_path):
    stop_words = set()
    stop_words.add(PIC_TOKEN)
    stop_words.add(VOICE_TOKEN)
    stop_words.add(LINE_TOKEN)
    if stopword_path:
        with open(stopword_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                stop_words.add(line)
    return stop_words


def save_data(data_train_x, data_train_target, data_test_x, train_x_path, train_y_path, test_x_path, stopwords_path=''):
    stopwords = read_stopwords(stopwords_path)
    with open(train_x_path, mode='w', encoding='utf-8') as f1, \
            open(train_y_path, mode='w', encoding='utf-8') as f2, \
            open(test_x_path, mode='w', encoding='utf-8') as f3:
        for line in data_train_x:
            if isinstance(line, str):
                seg_list = split_data(line.strip(), cut_type='word')
                seg_list = [word for word in seg_list if word not in stopwords]
                seg_line = ' '.join(seg_list)
                f1.write(seg_line)
            f1.write('\n')

        for line in data_train_target:
            if isinstance(line, str):
                seg_list = split_data(line.strip(), cut_type='word')
                seg_list = [word for word in seg_list if word not in stopwords]
                seg_line = ' '.join(seg_list)
                f2.write(seg_line)
            f2.write('\n')

        for line in data_test_x:
            if isinstance(line, str):
                seg_list = split_data(line.strip(), cut_type='word')
                seg_list = [word for word in seg_list if word not in stopwords]
                seg_line = ' '.join(seg_list)
                f3.write(seg_line)
            f3.write('\n')


if __name__ == '__main__':
    train_x, train_target = read_data(train_path)
    test_x, _ = read_data(test_path)
    save_data(train_x, train_target, test_x, train_seg_x_path, train_seg_target_path, test_seg_x_path)
