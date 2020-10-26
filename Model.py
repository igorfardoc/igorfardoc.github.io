import tensorflow as tf
L = tf.keras.layers
K = tf.keras.backend


def get_encoder(voc, emb_size, hid_size):
    emb = L.Embedding(len(voc), emb_size)
    rnn = L.GRU(units=hid_size, return_sequences=True, return_state=False, reset_after=False)

    input_tokens = L.Input([None], dtype='int32')
    input_embs = emb(input_tokens)
    states = rnn(input_embs)
    
    return tf.keras.Model(inputs=[input_tokens], outputs=[states])


def get_decoder_step(voc, emb_size, hid_size):
    # create layers
    emb = L.Embedding(len(voc), emb_size)
    rnn = L.GRU(units=hid_size, return_sequences=False, return_state=True, reset_after=False)
    token_prob = L.Dense(units=len(voc), activation='softmax')
    
    attention_from_enc = L.Dense(units=hid_size)
    attention_from_dec = L.Dense(units=hid_size)
    attention_second = L.Dense(units=1)
    

    # prev_tokens: [batch_size] of int32
    prev_tokens_in = L.Input([1], dtype='int32')
    # prev_state: [batch_size, hid_size], same as returned by encoder
    prev_state = L.Input([hid_size], dtype='float32')
    # states: encoder outputs [batch_size, length, hid_size]
    states = L.Input([None, hid_size], dtype='float32')
    prev_tokens = L.Reshape([])(prev_tokens_in)
    
    # apply attention
    
    enc_part = attention_from_enc(states)        # [batch_size, length, hid_size]
    dec_part = attention_from_dec(prev_state)    # [batch_size, hid_size]
    dec_part = L.Reshape([1, hid_size])(dec_part)      # [batch_size, 1, hid_size]
    
    after_first = L.Add()([enc_part, dec_part])  # [batch_size, length, hid_size]
    after_first = L.Activation('tanh')(after_first)
    
    after_second = attention_second(after_first) # -> [batch_size, length, 1]
    after_second = L.Reshape([-1])(after_second) # -> [batch_size, length]
    
    state_probs = L.Softmax(axis=1)(after_second) # -> [batch_size, length]
    
    summary_state = L.Dot(axes=(1, 1))([states, state_probs])  # [batch, hid_size]
    # end of attention
    token_embs = emb(prev_tokens)  # [batch_size, emb_size]
    rnn_inputs = L.Concatenate(axis=1)([token_embs, summary_state])
    rnn_inputs = L.Reshape([1, hid_size + emb_size])(rnn_inputs) # [batch_size, 1, hid_size + emb_size]
    new_state, _ = rnn(rnn_inputs, prev_state)
    # ^-- [batch_size, hid_size]

    new_token_probs = token_prob(new_state)
    # ^-- [batch_size, len(voc)]
    
    return tf.keras.Model(inputs=[prev_tokens_in, prev_state, states],
                          outputs=[new_state, new_token_probs, state_probs])
