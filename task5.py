import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
import string
import random
import os

def preprocess_text(text):
    """Preprocesses the text by converting to lowercase and removing unwanted characters."""
    text = text.lower()
    allowed_chars = string.ascii_lowercase + string.digits + " .,!?\n"
    text = ''.join(c for c in text if c in allowed_chars)
    return text

def create_char_mappings(text):
    """Creates character-to-index and index-to-character mappings."""
    chars = sorted(list(set(text)))
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_index, index_to_char

def create_sequences(text, seq_length, char_to_index):
    """Creates input-output sequences for training."""
    input_chars = []
    output_chars = []
    for i in range(0, len(text) - seq_length):
        input_seq = text[i:i + seq_length]
        output_seq = text[i + seq_length]
        input_chars.append([char_to_index[char] for char in input_seq])
        output_chars.append(char_to_index[output_seq])
    return np.array(input_chars), np.array(output_chars)

def create_model(vocab_size, seq_length):
    """Creates the RNN model."""
    model = Sequential([
        LSTM(256, input_shape=(seq_length, 1), return_sequences=True),
        LSTM(256),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

def generate_text(model, seed_text, seq_length, num_chars_to_generate, char_to_index, index_to_char):
    """Generates text from the trained model."""
    generated_text = seed_text
    for _ in range(num_chars_to_generate):
        input_eval = [char_to_index[char] for char in seed_text]
        input_eval = np.expand_dims(input_eval, axis=0)
        input_eval = np.expand_dims(input_eval, axis=-1) # add dimension for LSTM input

        predictions = model.predict(input_eval, verbose=0)
        predicted_id = np.random.choice(len(predictions[0]), p=predictions[0]) #Sample from the distribution.
        predicted_char = index_to_char[predicted_id]

        generated_text += predicted_char
        seed_text = seed_text[1:] + predicted_char

    return generated_text

def train_and_generate(file_path, seq_length=100, epochs=20, generate_length=500):
    """Trains the model and generates text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    text = preprocess_text(text)
    chars, char_to_index, index_to_char = create_char_mappings(text)
    vocab_size = len(chars)
    input_chars, output_chars = create_sequences(text, seq_length, char_to_index)

    input_chars = np.expand_dims(input_chars, axis=-1) # add dimension for LSTM input

    model = create_model(vocab_size, seq_length)
    model.fit(input_chars, output_chars, epochs=epochs, batch_size=128)

    start_index = random.randint(0, len(text) - seq_length - 1)
    seed_text = text[start_index:start_index + seq_length]
    generated_text = generate_text(model, seed_text, seq_length, generate_length, char_to_index, index_to_char)

    print("\nGenerated Text:\n", generated_text)

# Example Usage
# Create a text file named 'handwriting.txt' with sample handwritten-like text.
# For example, you can use a text file containing paragraphs of text.
# Then run:

if __name__ == "__main__":
    # Create a dummy text file if one does not exist.
    if not os.path.exists("handwriting.txt"):
      with open("handwriting.txt", "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. This is a sample text for handwriting generation. Hello world.")

    train_and_generate("handwriting.txt")