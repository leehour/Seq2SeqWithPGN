import time

import tensorflow as tf
from gensim.models import KeyedVectors
from tqdm import tqdm

from batch import batcher, article_to_ids, get_dec_inp_targ_seqs
from config import params, vocab_path, w2v_bin_path, train_seg_x_path, train_seg_target_path, test_seg_x_path
from entity.vocab import Vocab, START_DECODING, STOP_DECODING
from pgn_model import PGN
from test_helper import beam_decode
from utils.embedding_gen import get_embedding_pgn


def train_seq2seq(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    tf.compat.v1.logging.info("Loading the word2vec model ...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    vocab = Vocab(vocab_path, params['vocab_size'])
    encoder_embedding, decoder_embedding = get_embedding_pgn(vocab, train_seg_x_path, train_seg_target_path, w2v_model,
                                                             params['embedding_dim'])

    tf.compat.v1.logging.info("Building the model ...")
    model = PGN(params, encoder_embedding, decoder_embedding)

    tf.compat.v1.logging.info("Creating the batcher ...")
    datasets = batcher(train_seg_x_path, train_seg_target_path, vocab_path, params)

    tf.compat.v1.logging.info("Creating the checkpoint manager")
    logdir = "{}/logdir".format(params["model_dir"])
    checkpoint_dir = "{}\checkpoint".format(params["model_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    tf.compat.v1.logging.info("Starting the training ...")
    train_model(model, datasets, params, ckpt, ckpt_manager)


def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 1))
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_sum(loss_,
                              axis=-1) / dec_lens  # we have to make sure no empty abstract is being used otherwise dec_lens may contain null values
        return tf.reduce_mean(loss_)

    @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[], dtype=tf.int32)))
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len):
        loss = 0

        with tf.GradientTape() as tape:
            enc_hidden, enc_output = model.call_encoder(enc_inp)
            predictions, _ = model(enc_output, enc_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len)
            loss = loss_function(dec_tar, predictions)

        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables + model.pointer.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    try:
        for batch in dataset:
            # print("batch is {}".format(batch))
            t0 = time.time()
            loss = train_step(batch[0]["enc_input"], batch[0]["extended_enc_input"], batch[1]["dec_input"],
                              batch[1]["dec_target"], batch[0]["max_oov_len"])
            print('Step {}, time {:.4f}, Loss {:.4f}'.format(int(ckpt.step),
                                                             time.time() - t0,
                                                             loss.numpy()))
            if int(ckpt.step) == params["max_steps"]:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
                break
            if int(ckpt.step) % params["checkpoints_save_steps"] == 0:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
            ckpt.step.assign_add(1)
            break
    except KeyboardInterrupt:
        ckpt_manager.save(int(ckpt.step))
        print("Saved checkpoint for step {}".format(int(ckpt.step)))

    get_test_pred(model, test_seg_x_path, params)


def get_test_pred(model, filename, parmas):
    dataset = tf.data.TextLineDataset(filename)

    vocab = Vocab(vocab_path, parmas["vocab_size"])
    # print('vocab is {}'.format(vocab.word2id))

    for raw_record in dataset:
        article = raw_record.numpy().decode("utf-8")

        start_decoding = vocab.word_to_id(START_DECODING)
        stop_decoding = vocab.word_to_id(STOP_DECODING)

        article_words = article.split()[:parmas["max_enc_len"]]
        # print('article_words is {}'.format(article_words))
        enc_len = len(article_words)
        enc_input = [vocab.word_to_id(w) for w in article_words]
        enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)
        # print('enc_input_extend_vocab is {}'.format(enc_input_extend_vocab))
        # print('article_oovs is {}'.format(article_oovs))

        dec_input = [start_decoding]

        # inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
        #                                                        maxlen=max_length_inp,
        #                                                        padding='post')
        enc_input = tf.convert_to_tensor(enc_input)
        dec_input = tf.convert_to_tensor(dec_input)
        article_oovs = tf.convert_to_tensor(article_oovs)
        enc_input_extend_vocab = tf.convert_to_tensor(enc_input_extend_vocab)
        enc_input = tf.reshape(enc_input, (1, -1))
        # dec_input = tf.reshape(dec_input, (1, -1))
        article_oovs = tf.reshape(article_oovs, (1, -1))
        enc_input_extend_vocab = tf.reshape(enc_input_extend_vocab, (1, -1))

        enc_hidden, enc_output = model.call_encoder(enc_input, test_model=True)

        result = model.evaluate(enc_output, enc_hidden, enc_input_extend_vocab, dec_input,
                                        tf.shape(article_oovs)[1], params["max_dec_len"], vocab)

        print(result)


def test_model(params):
    # assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    vocab = Vocab(vocab_path, params['vocab_size'])
    encoder_embedding, decoder_embedding = get_embedding_pgn(vocab, train_seg_x_path, train_seg_target_path, w2v_model,
                                                             params['embedding_dim'])
    model = PGN(params, encoder_embedding, decoder_embedding)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    datasets = batcher(train_seg_x_path, train_seg_target_path, vocab_path, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}\checkpoint".format(params["model_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")

    for batch in datasets:
        yield beam_decode(model, batch, vocab, params)


def test_and_save(params):
    # assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test_model(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w") as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


if __name__ == '__main__':
    # train_seq2seq(params)
    # test_model(params)
    # test_and_save(params)
    tf.compat.v1.logging.info("Loading the word2vec model ...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    vocab = Vocab(vocab_path, params['vocab_size'])
    encoder_embedding, decoder_embedding = get_embedding_pgn(vocab, train_seg_x_path, train_seg_target_path, w2v_model,
                                                             params['embedding_dim'])

    tf.compat.v1.logging.info("Building the model ...")
    model = PGN(params, encoder_embedding, decoder_embedding)
    get_test_pred(model, test_seg_x_path, params)
