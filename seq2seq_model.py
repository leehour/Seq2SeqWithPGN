import tensorflow as tf


class Encoder(tf.keras.Model):
    """
    Seq2Seq Encoder
    """

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix, use_bigru=False):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        self.use_bigru = use_bigru
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        # self.gru = tf.keras.layers.GRU(self.enc_units,
        #                                return_sequences=True,
        #                                return_state=True,
        #                                recurrent_initializer='glorot_uniform')
        if use_bigru:
            self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, sentences, hidden):
        embed = self.embedding(sentences)
        if self.use_bigru:
            hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
            output, forward_state, backward_state = self.bigru(embed, initial_state=hidden)
            state = tf.concat([forward_state, backward_state], axis=1)
        else:
            output, state = self.gru(embed, initial_state=hidden)

        return output, state

    def initialize_hidden_state(self, test_model):
        if test_model:
            if self.use_bigru:
                return tf.zeros((1, 2 * self.enc_units))
            else:
                return tf.zeros((1, self.enc_units))
        else:
            if self.use_bigru:
                return tf.zeros((self.batch_sz, 2 * self.enc_units))
            else:
                return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    """
    LuongAttention
    """

    # other attention is LuongAttention
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape = (batch_size, hidden size)
        # values shape = (batch_size, max_length, hidden size)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, tf.squeeze(attention_weights, -1)


class Decoder(tf.keras.Model):
    """
    Seq2Seq Decoder
    """

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix, use_bigru=False):

        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.use_bigru = use_bigru
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.dec_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        # self.gru = tf.keras.layers.GRU(self.dec_units,
        #                                return_sequences=True,
        #                                return_state=True,
        #                                recurrent_initializer='glorot_uniform')
        if use_bigru:
            self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attentionEpoch
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, context_vector, is_train=True, dropout=False):

        # enc_output shape == (batch_size, max_length, hidden_size)
        # context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        if self.use_bigru:
            hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
            output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
            state = tf.concat([forward_state, backward_state], axis=1)
        else:
            # output shape == (batch_size * 1, 1, hidden_size)
            # state shape == (batch_size, hidden_size)
            # passing the concatenated vector to the GRU
            output, state = self.gru(x)
            # output shape == (batch_size, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        output = self.fc(output)
        if is_train and dropout:
            output = tf.nn.dropout(output, 0.5)

        return x, output, state


class Pointer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(state) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))
