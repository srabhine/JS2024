

sequence_input = Input(shape=(seq_length, 1), name='sequence_input')
embedding_size = 10  # Size of embedding vector
symbol_embedding = Embedding(input_dim=len(label_encoder.classes_), output_dim=embedding_size)(symbol_input)
lstm_out = LSTM(50, activation='relu')(sequence_input)
concat = Concatenate()([lstm_out, tf.squeeze(symbol_embedding, axis=1)]) # [N, dim]

symbol_out = []
for i in range(39):
    target_symbol_input = tf.where(tf.equal(input_tensor["symbol_id"], tf.cast(tf.ones_like(input_tensor["symbol_id"])*i, tf.string)), concat, tf.zeros_like(concat))
    target_symbol_out = Dense(units=32, activation='swish', name="symbol_"+str(i))(target_symbol_input) # [N, 32]
    symbol_out.append(target_symbol_out)
symbol_out_combined = tf.stack(symbol_out, axis=-1) # [N, 32, 1]
symbol_out_combined = tf.reduce_sum(symbol_out_combined, axis=-1) # [N ,32]
concat = symbol_out_combined

output = Dense(1)(concat)
model = Model(inputs=[sequence_input, symbol_input], outputs=output)
# n