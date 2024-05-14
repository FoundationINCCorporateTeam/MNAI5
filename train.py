# train.py
import numpy as np
from preprocessing import load_data, preprocess_data
from model import build_model

# Load and preprocess data
df = load_data()
tokenizer, max_len, X_train, X_test, y_train, y_test = preprocess_data(df)

# Model parameters
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1

# Build and train model
model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = build_model(vocab_size, embedding_dim, units, max_len)
y_train = np.expand_dims(y_train, -1)
y_test = np.expand_dims(y_test, -1)

model.fit([X_train, X_train], y_train, epochs=50, batch_size=64, validation_data=([X_test, X_test], y_test))

# Save model and tokenizer
model.save('seq2seq_chatbot_model.h5')
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
