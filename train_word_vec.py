from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

from config import w2v_bin_path, \
    train_seg_x_path, train_seg_target_path, test_seg_x_path, w2v_output_path, \
    sentences_path, embedding_size
from utils.data_utils import dump_pkl, load_pkl


def read_lines(path):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def extract_sentence(train_seg_x_path, train_seg_target_path, test_seg_x_path, col_sep='\t'):
    ret = []
    print('read %s...' % train_seg_x_path)
    lines = read_lines(train_seg_x_path)
    print('read %s...' % train_seg_target_path)
    lines += read_lines(train_seg_target_path)
    print('read %s...' % test_seg_x_path)
    lines += read_lines(test_seg_x_path)
    for line in lines:
        ret.append(line)
        # if col_sep in line:
        #     index = line.index(col_sep)
        #     word_tag = line[index + 1:]
        #     sentence = ''.join(get_sentence(word_tag))
        #     ret.append(sentence)
    return ret


def save_sentence(sentences, sentence_path):
    print('start to save sentence:%s' % sentence_path)
    with open(sentence_path, mode='w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write('%s\n' % sentence)
    print('finish save sentence:%s' % sentence_path)


def build(train_seg_x_path, train_seg_target_path, test_seg_x_path, w2v_output, sentence_path,
          w2v_bin_path="model.bin", embedding_size=256, min_count=5, col_sep='\t'):
    # sentences = extract_sentence(train_seg_x_path, train_seg_target_path, test_seg_x_path, col_sep=col_sep)
    # save_sentence(sentences, sentence_path)
    #
    # print('train w2v model...')
    # # train model
    # w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
    #                size=embedding_size, window=5, min_count=min_count, iter=40)
    # w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    # print("save %s ok." % w2v_bin_path)
    # # test
    # sim = w2v.wv.similarity('奔驰', '宝马')
    # print('奔驰 vs 宝马 similarity score:', sim)

    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word2vec_dict = {}
    for word in model.vocab:
        word2vec_dict[word] = model.word_vec(word)
    dump_pkl(word2vec_dict, w2v_output, True)


if __name__ == '__main__':
    build(train_seg_x_path, train_seg_target_path, test_seg_x_path, w2v_output_path, sentences_path,
          w2v_bin_path=w2v_bin_path, embedding_size=embedding_size, min_count=5)
