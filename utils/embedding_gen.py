import numpy as np


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
