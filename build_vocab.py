from collections import defaultdict

from config import sentences_path, w2v_bin_path, vocab_path
from utils.data_utils import read_lines, save_word_dict


def generate_vocab(sentence_path, min_count=0, lower=False, sort=True):
    sentences = read_lines(sentence_path)

    word_dict = defaultdict(int)
    for sentence in sentences:
        sentence = sentence if lower else sentence.lower()
        for word in sentence.split():
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    if sort:
        word_tulpe = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
        result = [word for i, (word, count) in enumerate(word_tulpe) if count >= min_count]
    else:
        result = [word for word, count in word_dict.items() if count >= min_count]
    word_ids = dict([(word, index) for index, word in enumerate(result)])
    ids_word = dict([(index, word) for index, word in enumerate(result)])
    return word_ids, ids_word


if __name__ == '__main__':
    vocab, reverse_vocab = generate_vocab(sentences_path)
    save_word_dict(vocab, vocab_path)
