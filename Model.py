import tensorflow as tf
L = tf.keras.layers
K = tf.keras.backend


def get_encoder(voc, emb_size, hid_size):
    emb = L.Embedding(len(voc), emb_size)
    rnn = L.GRU(units=hid_size, return_sequences=True, return_state=False)

    input_tokens = L.Input([None], dtype='int32')
    input_embs = emb(input_tokens)
    states = rnn(input_embs)
    
    return tf.keras.Model(inputs=[input_tokens], outputs=[states])


def get_decoder_step(voc, emb_size, hid_size):
    # create layers
    emb = L.Embedding(len(voc), emb_size)
    rnn = L.GRU(units=hid_size, return_sequences=False, return_state=True)
    token_prob = L.Dense(units=len(voc), activation='softmax')
    attention_first = L.Dense(units=123, activation='tanh')
    attention_second = L.Dense(units=1)
    

    # prev_tokens: [batch_size] of int32
    prev_tokens = L.Input([], dtype='int32')
    # prev_state: [batch_size, hid_size], same as returned by encoder
    prev_state = L.Input([hid_size], dtype='float32')
    # states: encoder outputs [batch_size, length, hid_size]
    states = L.Input([None, hid_size], dtype='float32')

    
    # apply attention
    length = K.shape(states)[1]
    batch_size = K.shape(states)[0]
    dec_states = K.tile(K.reshape(prev_state, [batch_size, 1, -1]), [1, length, 1])
    concat = K.concatenate([states, dec_states], axis=2)
    concat.set_shape([None, None, 2 * hid_size])
    
    after_first = attention_first(concat) # -> [batch_size, SIZE, hid_size]
    after_second = attention_second(after_first) # -> [batch_size, SIZE, 1]
    state_probs = K.softmax(after_second, axis=1) # -> [batch_size, SIZE, 1]
    summary_state = K.sum(states * state_probs, axis=1)
    # end of attention
    
    token_embs = emb(prev_tokens)  # [batch_size, emb_size]
    rnn_inputs = K.expand_dims(K.concatenate([token_embs, summary_state], axis=1), axis=1)
    
    new_state, _ = rnn(rnn_inputs, prev_state)
    # ^-- [batch_size, hid_size]

    new_token_probs = token_prob(new_state)
    # ^-- [batch_size, len(voc)]
    
    return tf.keras.Model(inputs=[prev_tokens, prev_state, states],
                          outputs=[new_state, new_token_probs, state_probs])
