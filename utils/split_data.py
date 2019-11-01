from jieba import posseg
import jieba


def split_data(sentence, cut_type='word', pos=False):
    if pos:
        pos_seq = []
        if cut_type == 'word':
            word_seq = []
            word_cut = posseg.lcut(sentence)
            for word, pos in word_cut:
                word_seq.append(word)
                pos_seq.append(pos)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            for word in word_seq:
                pos_s = posseg.lcut(word)
                pos_seq.append(pos_s[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


if __name__ == '__main__':
    print(split_data('对于性能有更高的要求时,有时我们的服务需要传递大量的数据', cut_type='char', pos=True))
