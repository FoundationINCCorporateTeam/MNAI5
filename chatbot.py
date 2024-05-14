# chatbot.py
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data
from model import build_inference_models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('seq2seq_chatbot_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Model parameters
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
max_len = 20  # Same max_len used during training

# Rebuild inference models
encoder_inputs = model.input[0]
decoder_inputs = model.input[1]
encoder_states = model.layers[4].output
decoder_embedding = model.layers[3](decoder_inputs)
decoder_lstm = model.layers[5]
decoder_dense = model.layers[6]

encoder_model, decoder_model = build_inference_models(vocab_size, embedding_dim, units, max_len, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['startseq']

    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'endseq' or len(decoded_sentence.split()) > max_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    return decoded_sentence.strip()

# Console interaction
print("Chatbot: Hi! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    
    input_seq = tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    response = decode_sequence(input_seq)
    
    print(f"Chatbot: {response}")
