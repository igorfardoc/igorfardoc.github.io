import tensorflow as tf
L = tf.keras.layers


class Encoder(L.Layer):
    def __init__(self, voc, emb_size, hid_size, out_size):
        super().__init__()
        self.voc, self.emb_size, self.hid_size, self.out_size = voc, emb_size, hid_size, out_size
        
        self.emb = L.Embedding(len(voc), emb_size)
        
        self.rnn = L.GRUCell(units=hid_size, activation='tanh')
        self.rnn.build([emb_size])
        
        
    def call(self, tokens_tensor):
        batch_size, max_length = tokens_tensor.shape
        state = tf.zeros([batch_size, self.hid_size])
        states_list = []
        for t in range(tokens_tensor.shape[1]):
            token_embs = self.emb(tokens_tensor[:, t])      # [batch_size, emb_size]
            new_state, _ = self.rnn(token_embs, state)      # [batch_size, hid_siize]
            states_list.append(tf.reshape(new_state, [batch_size, 1, -1]))
            state = new_state
        states = tf.concat(states_list, axis=1)
        return states


class Attention(L.Layer):
    def __init__(self, hid_size):
        super().__init__()
        self.first_layer = L.Dense(units=hid_size, activation='tanh')
        self.first_layer.build([2 * hid_size])
        self.second_layer = L.Dense(units=1)
        self.second_layer.build([hid_size])
        
    def call(self, dec_state, states):
        # dec_state -> [batch_size, hid_size]
        # states -> [batch_size, SIZE, hid_size]
        # summary_state -> [batch_size, hid_size]
        # state_probs -> [batch_size, SIZE]
        length = states.shape[1]
        batch_size = states.shape[0]
        dec_states = tf.tile(tf.reshape(dec_state, [batch_size, 1, -1]), [1, length, 1])
        after_first = self.first_layer(tf.concat([states, dec_states], axis=2)) # -> [batch_size, SIZE, hid_size]
        after_second = self.second_layer(after_first) # -> [batch_size, SIZE, 1]
        state_probs = tf.nn.softmax(after_second, axis=1) # -> [batch_size, SIZE, 1]
        summary_state = tf.reduce_sum(states * state_probs, axis=1)
        return summary_state, tf.reshape(state_probs, [batch_size, -1])
    
        
class DecoderStep(L.Layer):
    def __init__(self, voc, emb_size, hid_size):
        super().__init__()
        self.voc, self.emb_size, self.hid_size = voc, emb_size, hid_size
        
        # TODO: create token embeddings for german; recurrent cell; layer that predicts next token probabilities
        self.emb = L.Embedding(len(voc), emb_size)
        self.rnn = L.GRUCell(units=hid_size)
        self.rnn.build([emb_size + hid_size])
        self.token_prob = L.Dense(units=len(voc), activation='softmax')
        self.token_prob.build([hid_size])
        
        self.attention = Attention(hid_size)
        
        
    def call(self, prev_tokens, prev_state, states):
        # prev_tokens: [batch_size] of int32
        # prev_state: [batch_size, hid_size], same as returned by encoder
        batch_size = states.shape[0]
        token_embs = self.emb(prev_tokens)
        summary_state, state_probs = self.attention(prev_state, states)
        new_state, _ = self.rnn(tf.concat([token_embs, summary_state], axis=1), prev_state)
        # ^-- [batch_size, hid_size]
        
        new_token_probs = self.token_prob(new_state)
        # ^-- [batch_size, len(voc)]
        
        return new_state, new_token_probs, state_probs