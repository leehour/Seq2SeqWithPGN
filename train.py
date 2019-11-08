import time

import tensorflow as tf
from gensim.models import KeyedVectors

from batch import batcher
from config import params, vocab_path, w2v_bin_path, train_seg_x_path, train_seg_target_path
from entity.vocab import Vocab
from pgn_model import PGN
from utils.embedding_gen import get_embedding_pgn


def train_seq2seq(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    tf.compat.v1.logging.info("Loading the word2vec model ...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    vocab = Vocab(vocab_path, params['vocab_size'])
    encoder_embedding, decoder_embedding = get_embedding_pgn(vocab, train_seg_x_path, train_seg_target_path, w2v_model,
                                                             params['embedding_dim'])
    print(encoder_embedding.shape)
    print(decoder_embedding.shape)

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
        loss_ = tf.reduce_sum(loss_, axis=-1) / dec_lens  # we have to make sure no empty abstract is being used otherwise dec_lens may contain null values
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
            if int(ckpt.step) % params["checkpoints_save_steps"] == 0:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
            ckpt.step.assign_add(1)
    except KeyboardInterrupt:
        ckpt_manager.save(int(ckpt.step))
        print("Saved checkpoint for step {}".format(int(ckpt.step)))


if __name__ == '__main__':
    train_seq2seq(params)
