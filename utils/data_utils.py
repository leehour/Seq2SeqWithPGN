import os
import pickle


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def read_lines(path):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def save_word_dict(dict_data, vocab_path):
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write('%s\t%d\n' % (k, v))


def save_result(file_path, line):
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(len(line)):
            f.write('%s\n' % line[i])
