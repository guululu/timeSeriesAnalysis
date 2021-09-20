import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, GlobalAveragePooling1D, Lambda, \
    Layer, Dot


class BahdanauAttention(Layer):

    '''
    Bahdanau's attention mechanis,
    '''

    def __init__(self, units, verbose=0):

        '''
        Here we consider the general scoring function

        Args:
            units: dimensionality of the context vector
            verbo
        '''

        super(BahdanauAttention, self).__init__()

        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.verbose = verbose

    def call(self, query, values):

        if self.verbose:
            print('\n******* Bahdanau Attention STARTS *******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ',
                  values.shape)

        query_with_time_axis = tf.expand_dims(query, 1)

        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        if self.verbose:
            print('score: (batch_size, max_length, 1)', score.shape)

        attention_weights = tf.nn.softmax(score, axis=1)

        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1)', attention_weights.shape)

        context_vector = attention_weights * values

        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ', context_vector.shape)

        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size)', context_vector.shape)

        context_vector = tf.reduce_sum(context_vector, axis=1)
        print('\n******* Bahdanau Attention ENDS ******')

        return context_vector, attention_weights


class LuongAttention(Layer):

    '''
    Luong's attention mechanis,
    '''

    def __init__(self, units, verbose=0):

        '''
        Here we consider the general scoring function

        Args:
            units: dimensionality of the context vector
            verbose:
        '''

        super(LuongAttention, self).__init__()

        self.W = Dense(units)
        self.verbose = verbose

    def call(self, query, values):

        if self.verbose:
            print('\n******* Luong Attention STARTS *******')
            print('query (decoder hidden state): (batch_size, 1, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ',
                  values.shape)

        # query_with_time_axis = tf.expand_dims(query, 1)
        #
        # if self.verbose:
        #     print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        adjusted_values = self.W(values)
        adjusted_values_transposed = tf.transpose(adjusted_values, perm=[0, 2, 1])
        if self.verbose:
            print('adjusted_values:(batch_size, hidden size, max_len) ', adjusted_values_transposed.shape)

        score = tf.transpose(tf.matmul(query, adjusted_values_transposed), perm=[0, 2, 1])

        if self.verbose:
            print('score: (batch_size, max_length, 1)', score.shape)

        attention_weights = tf.nn.softmax(score, axis=1)

        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1)', attention_weights.shape)

        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ', context_vector.shape)

        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size)', context_vector.shape)

        print('\n******* Luong Attention ENDS ******')

        return context_vector, attention_weights