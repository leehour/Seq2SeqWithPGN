import tensorflow as tf

from config import vocab_path, train_seg_x_path, train_seg_target_path, params
from entity.vocab import UNKNOWN_TOKEN, SENTENCE_START, SENTENCE_END, Vocab, START_DECODING, STOP_DECODING


def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def output_to_words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id_to_word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract_to_sents(abstract):
    """
    Splits abstract text from datafile into list of sentences.
    Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences
    Returns:
    sents: List of sentence strings (no tags)
    """
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START): end_p])
        except ValueError as e:  # no more sentences
            return sents


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder,
    and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len.
    The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


# def _parse_function(example_proto):
#     # Create a description of the features.
#     feature_description = {'article': tf.io.FixedLenFeature([], tf.string, default_value=''),
#                            'abstract': tf.io.FixedLenFeature([], tf.string, default_value='')}
#     # Parse the input `tf.Example` proto using the dictionary above.
#     parsed_example = tf.io.parse_single_example(example_proto, feature_description)
#     return parsed_example


def example_generator(filenames_1, filenames_2, vocab_path, vocab_size, max_enc_len, max_dec_len, mode):
    dataset_1 = tf.data.TextLineDataset(filenames_1)
    dataset_2 = tf.data.TextLineDataset(filenames_2)

    print(dataset_1)
    train_dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
    if mode == "train":
        train_dataset = train_dataset.shuffle(10, reshuffle_each_iteration=True).repeat()

    vocab = Vocab(vocab_path, vocab_size)
    # print('vocab is {}'.format(vocab.word2id))

    for raw_record in train_dataset:
        article = raw_record[0].numpy().decode("utf-8")
        abstract = raw_record[1].numpy().decode("utf-8")

        start_decoding = vocab.word_to_id(START_DECODING)
        stop_decoding = vocab.word_to_id(STOP_DECODING)

        article_words = article.split()[:max_enc_len]
        # print('article_words is {}'.format(article_words))
        enc_len = len(article_words)
        enc_input = [vocab.word_to_id(w) for w in article_words]
        enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)
        # print('enc_input_extend_vocab is {}'.format(enc_input_extend_vocab))
        # print('article_oovs is {}'.format(article_oovs))

        abstract_sentences = [sent.strip() for sent in abstract_to_sents(abstract)]
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word_to_id(w) for w in abstract_words]
        abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
        dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
        _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)
        dec_len = len(dec_input)

        output = {
            "enc_len": enc_len,
            "enc_input": enc_input,
            "enc_input_extend_vocab": enc_input_extend_vocab,
            "article_oovs": article_oovs,
            "dec_input": dec_input,
            "target": target,
            "dec_len": dec_len,
            "article": article,
            "abstract": abstract,
            "abstract_sents": abstract_sentences
        }
        yield output


def batch_generator(generator, filenames_1, filenames_2, vocab_path, vocab_size, max_enc_len, max_dec_len, batch_size,
                    mode):
    dataset = tf.data.Dataset.from_generator(generator,
                                             args=[filenames_1, filenames_2, vocab_path, vocab_size, max_enc_len,
                                                   max_dec_len, mode],
                                             output_types={
                                                 "enc_len": tf.int32,
                                                 "enc_input": tf.int32,
                                                 "enc_input_extend_vocab": tf.int32,
                                                 "article_oovs": tf.string,
                                                 "dec_input": tf.int32,
                                                 "target": tf.int32,
                                                 "dec_len": tf.int32,
                                                 "article": tf.string,
                                                 "abstract": tf.string,
                                                 "abstract_sents": tf.string
                                             },
                                             output_shapes={
                                                 "enc_len": [],
                                                 "enc_input": [None],
                                                 "enc_input_extend_vocab": [None],
                                                 "article_oovs": [None],
                                                 "dec_input": [None],
                                                 "target": [None],
                                                 "dec_len": [],
                                                 "article": [],
                                                 "abstract": [],
                                                 "abstract_sents": [None]
                                             })

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({"enc_len": [],
                                                   "enc_input": [None],
                                                   "enc_input_extend_vocab": [None],
                                                   "article_oovs": [None],
                                                   "dec_input": [max_dec_len],
                                                   "target": [max_dec_len],
                                                   "dec_len": [],
                                                   "article": [],
                                                   "abstract": [],
                                                   "abstract_sents": [None]}),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": 1,
                                                   "enc_input_extend_vocab": 1,
                                                   "article_oovs": b'',
                                                   "dec_input": 1,
                                                   "target": 1,
                                                   "dec_len": -1,
                                                   "article": b"",
                                                   "abstract": b"",
                                                   "abstract_sents": b''},
                                   drop_remainder=True)

    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"]})

    dataset = dataset.map(update)
    return dataset


def batcher(filenames_1, filenames_2, vocab_path, parmas):
    # filenames = glob.glob("{}/*.tfrecords".format(data_path))
    dataset = batch_generator(example_generator, filenames_1, filenames_2, vocab_path, parmas["vocab_size"],
                              parmas["max_enc_len"],
                              parmas["max_dec_len"], parmas["batch_size"], parmas["mode"])
    return dataset


if __name__ == '__main__':
    dataset = batcher(train_seg_x_path, train_seg_target_path, vocab_path, params)
    # train_seq2seq(dataset, params)
