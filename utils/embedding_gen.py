import os
from collections import defaultdict

import numpy as np

from config import train_seg_x_path, train_seg_target_path, test_seg_x_path
from entity.vocab import UNKNOWN_TOKEN
from utils.data_utils import read_lines


def get_embedding(word_index_input, word_index_target, model, embed_size=256):
    """
    获取input和target的embedding matrix
    :param word_index_input: 输入词典
    :param word_index_target: 输出词典
    :param embed_size:embedding size
    :return:
    """
    num_input_en = len(word_index_input) + 1
    # num_input_en = min(max_words_size, len(word_index_input))
    encoder_embedding = np.zeros((num_input_en, embed_size))
    for word, id in word_index_input.items():
        if id < num_input_en:
            if word not in model.vocab:
                word_vec = np.random.uniform(-0.25, 0.25, embed_size)
            else:
                word_vec = model.word_vec(word)
            encoder_embedding[id] = word_vec

    num_input_de = len(word_index_target) + 1
    # num_input_de = min(max_words_size, len(word_index_target))
    decoder_embedding = np.zeros((num_input_de, embed_size))
    for word, id in word_index_target.items():
        if id < num_input_de:
            if word not in model.vocab:
                word_vec = np.random.uniform(-0.25, 0.25, embed_size)
            else:
                word_vec = model.word_vec(word)
            decoder_embedding[id] = word_vec
    return encoder_embedding, decoder_embedding


def get_text_vocab(train_seg_x_path, train_seg_target_path):
    ret = []
    print('read %s...' % train_seg_x_path)
    lines_train = read_lines(train_seg_x_path)
    print('read %s...' % train_seg_target_path)
    lines_target = read_lines(train_seg_target_path)

    train_word_set = set()
    for sentence in lines_train:
        for word in sentence.split():
            if word not in train_word_set:
                train_word_set.add(word)

    target_word_set = set()
    for sentence in lines_target:
        for word in sentence.split():
            if word not in target_word_set:
                target_word_set.add(word)
    return train_word_set, target_word_set




def get_embedding_pgn(vocab, train_seg_x_path, train_seg_target_path, model, embed_size=256):
    train_word, target_word = get_text_vocab(train_seg_x_path, train_seg_target_path)

    num_input_en = vocab.size() + 1
    # num_input_en = min(max_words_size, len(word_index_input))
    encoder_embedding = np.zeros((num_input_en, embed_size))
    for word in train_word:
        word_id_en = vocab.word_to_id(word)
        if word_id_en == UNKNOWN_TOKEN or word not in model.vocab:
            word_vec = np.random.uniform(-0.25, 0.25, embed_size)
        else:
            word_vec = model.word_vec(word)
        encoder_embedding[word_id_en] = word_vec

    num_input_de = vocab.size() + 1
    # num_input_de = min(max_words_size, len(word_index_target))
    decoder_embedding = np.zeros((num_input_de, embed_size))
    for word in target_word:
        word_id_de = vocab.word_to_id(word)
        if word_id_de == UNKNOWN_TOKEN or word not in model.vocab:
            word_vec = np.random.uniform(-0.25, 0.25, embed_size)
        else:
            word_vec = model.word_vec(word)
        decoder_embedding[word_id_de] = word_vec
    return encoder_embedding, decoder_embedding

if __name__ == '__main__':
    train_seg_x_path = os.path.join(os.path.abspath('..'), 'datasets', 'train_seg_x.csv')
    # os.path.join(os.path.abspath('..'), '/datasets', 'train_seg_x.csv')
    train_seg_target_path = os.path.join(os.path.abspath('..'), 'datasets', 'train_seg_target.csv')
    train_word, target_word = get_text_vocab(train_seg_x_path, train_seg_target_path)
    for word in train_word:
        print(word)
        break