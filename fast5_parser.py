import numpy as np
import h5py as hp
import os
import re
import sys
import math
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.model import Sequential, Model
import numpy

def preprocess_signals(signals):
    """
    Preprocesses signals by slicing long ones to default length and zero-padding the short ones
    :param signals: numpy array of signals
    :return: numpy array of preprocessed signals
    """
    pass

def get_representations(signals):
    """
    Function builds and trains an autodecoder for representation learning
    :param signals: numpy array of signals #TODO think about appropriate dimensions
    :return: numpy array of representations #TODO same
    """
    lstm_model = Sequential()
    lstm_model.add(LSTM(1000, activation='relu'))
    lstm_model.add(RepeatVector(signals.shape[0]))
    lstm_model.add(LSTM(1000, activation='relu', return_sequences=True))
    lstm_model.add(TimeDistributed(Dense(1)))

    lstm_model.compile(optimizer='adam', loss='mse')

    lstm_model.fit(signals, signals, epochs=50, verbose=0)

    lstm_model = Model(inputs=lstm_model.inputs, outputs=lstm_model.layers[0].output)

    representations = lstm_model.predict(signals)

def parse_file(entry):
    file = hp.File(entry, 'r')
    level1 = (file['Raw'])['Reads']
    final = []
    for key in level1.keys():
        if re.match('^Read_.*', key):
            level2 = (level1[key])['Signal']
            values = (level2[()])
            final.append(values)
    return final

def parse(path):
    """
    Parsing of .fast5 files containing signal values
    :param path: path to a folder or a single .fast5 containing signal values
    :return: 1D array where each value in array represents an array of signals for a certain .fast5 file
    """
    all_signals = []
    if re.match("^.*signal.*fast5", path):
        all_signals.append(parse_file(path))
    else:
        for entry in os.listdir(path):
            if re.match("^signal_.*fast5", entry):
                all_signals.append(parse_file(path + entry))
    return np.array(all_signals)

def len_statistics(signals):
    lens = np.array([len(signal) for file in signals for signal in file])
    find_most_common_length(lens)
    mean = np.mean(lens)
    std = math.sqrt(np.mean((lens - mean)**2))
    return mean, std

def plot_signals(signals):
    for file in signals:
        for signal in file:
            plt.plot(signal)
            plt.show()


def find_most_common_length(lengths):
    values, counts = np.unique(lengths, return_counts=True)
    for value, count in zip(values, counts):
        if count > 1:
            print("Length {} occurs {} times".format(value, count))

if __name__ == "__main__":
    path = sys.argv[1]
    all_signals = parse(path)
    #plot_signals(all_signals)
    len_mean, len_std = len_statistics(all_signals)
    print("The average length of signals is {} with standard deviation of {}".format(len_mean, len_std))


