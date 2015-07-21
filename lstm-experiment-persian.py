from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
import numpy as np
import random, sys
import pandas as pd

# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i

def main():
    verses = pd.read_csv('/home/hpc/Erfan/verse_table.csv', encoding='utf-8')
    a = [a for a in verses.text.tolist()[:100000] if type(a) == unicode]
    text = unicode('\n').join(a)
    print('corpus length:', len(text))
    
    chars = set(text)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 15
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i + maxlen])
    print('number of sequences:', len(sentences))
    
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(len(chars), 2048, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(2048, 1024, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(1024, 512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(512, len(chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # train the model, output generated text after each iteration
    # for iteration in range(1, 20):
    #     print()
    #     print('-' * 50)
    #     print('Iteration', iteration)
    model.fit(X, y, batch_size=1024, nb_epoch=100)
    model.save_weights('/home/hpc/Erfan/ipython/weights.big.final.hdf5', overwrite=True)
    
        # start_index = random.randint(0, len(a)-1)
    
        # for diversity in [0.2, 0.4, 0.6, 0.8]:
        #     print()
        #     print('----- diversity:', diversity)
    
        #     generated = ''
        #     sentence = a[start_index] + '\n'
        #     generated += sentence
        #     print('----- Generating with seed: "' + sentence + '"')
        #     sys.stdout.write(generated)
    
        #     for iteration in range(450):
        #         x = np.zeros((1, len(sentence), len(chars)))
        #         for t, char in enumerate(sentence):
        #             x[0, t, char_indices[char]] = 1.
    
        #         preds = model.predict(x, verbose=0)[0]
        #         next_index = sample(preds, diversity)
        #         next_char = indices_char[next_index]
    
        #         generated += next_char
        #         sentence = sentence[1:] + next_char
    
        #         sys.stdout.write(next_char)
        #         sys.stdout.flush()
        #     print()
        
if __name__ == '__main__':
    main()
