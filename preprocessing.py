# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data():
    def load_conversations(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        conversations = []
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                conversations.append(parts)
        return conversations

    def load_lines(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        id2line = {}
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                id2line[parts[0]] = parts[4]
        return id2line

    def extract_conversations(conversations, id2line):
        qa_pairs = []
        for conv in conversations:
            line_ids = conv[-1][1:-1].replace("'", "").replace(" ", "").split(',')
            for i in range(len(line_ids) - 1):
                qa_pairs.append((id2line[line_ids[i]], id2line[line_ids[i + 1]]))
        return qa_pairs

    lines_file = 'movie_lines.txt'
    conversations_file = 'movie_conversations.txt'

    id2line = load_lines(lines_file)
    conversations = load_conversations(conversations_file)
    qa_pairs = extract_conversations(conversations, id2line)

    return pd.DataFrame(qa_pairs, columns=['input', 'target'])

def preprocess_data(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['input'].tolist() + df['target'].tolist())

    input_sequences = tokenizer.texts_to_sequences(df['input'].tolist())
    target_sequences = tokenizer.texts_to_sequences(df['target'].tolist())

    max_len = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(input_sequences, target_sequences, test_size=0.2)

    return tokenizer, max_len, X_train, X_test, y_train, y_test
