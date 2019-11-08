SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '[UNK]'
PAD_TOKEN = '[PAD]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'


class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2id = {UNKNOWN_TOKEN: 0, PAD_TOKEN: 1, START_DECODING: 2, STOP_DECODING: 3}
        self.id2word = {0: UNKNOWN_TOKEN, 1: PAD_TOKEN, 2: START_DECODING, 3: STOP_DECODING}
        self.count = 4

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()

                if len(line) != 2:
                    print('Incorrect vocab dict pattern %s:' % line)
                    continue

                word = line[0]
                if word in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(r'Invalid word: %s' % word)

                if word in self.word2id:
                    continue

                self.word2id[word] = self.count
                self.id2word[self.count] = word
                self.count += 1

                if max_size != 0 and self.count >= max_size:
                    break

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, id):
        if id not in self.id2word:
            raise Exception(r'Id %d not Found in vocab ' % id)
        return self.id2word[id]

    def size(self):
        return self.count

