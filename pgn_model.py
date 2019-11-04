import tensorflow as tf

from seq2seq_model import Encoder, BahdanauAttention, Decoder, Pointer
from utils.calc_dist_utils import _calc_final_dist


class PGN(tf.keras.Model):
    def __init__(self, params, encoder_embedding_matrix, decoder_embedding_matrix):
        super().__init__()
        self.params = params
        self.encoder = Encoder(params['vocab_size'], params['embedding_dim'], params['enc_units'],
                               params['batch_size'], embedding_matrix=[encoder_embedding_matrix])
        self.attention = BahdanauAttention(params['attn_units'])
        self.decoder = Decoder(params['vocab_size'], params['embedding_dim'], params['dec_units'],
                               params['batch_size'], embedding_matrix=[decoder_embedding_matrix])
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len):
        predictions = []
        attentions = []
        p_gens = []
        context_vector, attention = self.attention(dec_hidden, enc_output)
        for i in range(dec_inp.shape[1]):
            dec_output, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:i], axis=1),
                                                  dec_hidden,
                                                  enc_output,
                                                  context_vector)
            context_vector, attention = self.attention(dec_hidden, enc_output)
            p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_inp, axis=1))
            predictions.append(dec_output)
            attentions.append(attention)
            p_gens.append(p_gen)
        final_dists = _calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len,
                                       self.params["vocab_size"], self.params["batch_size"])
        return tf.stack(final_dists, 1), dec_hidden


if __name__ == '__main__':
    encoder = Encoder(vocab_size=25216, embedding_dim=256, enc_units=1024, batch_sz=64)
    sample_hidden = encoder.initialize_hidden_state()
    example_input_batch = tf.ones(shape=(64, 88), dtype=tf.int32)
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(128)
    attention_weights, attention_result = attention_layer(sample_hidden, sample_output)
    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size=13053, embedding_dim=256, dec_units=1024, batch_sz=64)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
