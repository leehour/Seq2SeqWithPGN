import tensorflow as tf
from gensim.models import KeyedVectors

from config import vocab_path, params, w2v_bin_path, train_seg_x_path, train_seg_target_path
from entity.vocab import Vocab
from seq2seq_model import Encoder, BahdanauAttention, Decoder, Pointer
from utils.calc_dist_utils import _calc_final_dist
from utils.embedding_gen import get_embedding_pgn


class PGN(tf.keras.Model):
    def __init__(self, params, encoder_embedding_matrix, decoder_embedding_matrix, use_bigru=True):
        super().__init__()
        self.params = params
        self.encoder = Encoder(params['vocab_size'], params['embedding_dim'], params['enc_units'],
                               params['batch_size'], embedding_matrix=encoder_embedding_matrix, use_bigru=False)
        self.attention = BahdanauAttention(params['attn_units'])
        self.decoder = Decoder(params['vocab_size'], params['embedding_dim'], params['dec_units'],
                               params['batch_size'], embedding_matrix=decoder_embedding_matrix)
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_hidden, enc_output

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len):
        predictions = []
        attentions = []
        p_gens = []
        context_vector, _ = self.attention(dec_hidden, enc_output)
        for t in range(dec_inp.shape[1]):
            dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                   dec_hidden,
                                                   enc_output,
                                                   context_vector)
            context_vector, attn = self.attention(dec_hidden, enc_output)
            p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))

            predictions.append(pred)
            attentions.append(attn)
            p_gens.append(p_gen)

        final_dists = _calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len,
                                       self.params["vocab_size"], self.params["batch_size"])
        # predictions_shape = (batch_size, dec_len, vocab_size) with dec_len = 1 in pred mode
        return tf.stack(final_dists, 1), dec_hidden


if __name__ == '__main__':
    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    vocab = Vocab(vocab_path, params['vocab_size'])
    encoder_embedding, decoder_embedding = get_embedding_pgn(vocab, train_seg_x_path, train_seg_target_path, w2v_model,
                                                             params['embedding_dim'])

    encoder = Encoder(vocab_size=vocab.size(), embedding_dim=256, enc_units=256, batch_sz=32,
                      embedding_matrix=encoder_embedding)
    sample_hidden = encoder.initialize_hidden_state()
    example_input_batch = tf.ones(shape=(32, 88), dtype=tf.int32)
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(128)
    context_vector, attention_result = attention_layer(sample_hidden, sample_output)
    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_result.shape))

    decoder = Decoder(vocab_size=vocab.size(), embedding_dim=256, dec_units=256, batch_sz=32,
                      embedding_matrix=decoder_embedding)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((32, 1)), sample_hidden, sample_output, context_vector)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

    pgn = PGN(params, encoder_embedding, decoder_embedding)
    enc_hidden, enc_output = pgn.call_encoder(example_input_batch)
    predictions, _ = pgn(enc_output, sample_hidden, example_input_batch,
                         tf.random.uniform([32, 2], minval=1, maxval=10,dtype=tf.int32),tf.random.uniform((32, 1)), 6)

    print("finished")
