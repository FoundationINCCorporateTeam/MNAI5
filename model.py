# model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import Adam

def build_model(vocab_size, embedding_dim, units, max_len):
    # Encoder
    encoder_inputs = Input(shape=(max_len,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_len,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')

    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense

def build_inference_models(vocab_size, embedding_dim, units, max_len, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense):
    # Encoder model for inference
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder model for inference
    decoder_state_input_h = Input(shape=(units,))
    decoder_state_input_c = Input(shape=(units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model
