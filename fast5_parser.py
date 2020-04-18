import numpy as np
import h5py as hp
import os
import re
import sys
import math
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_train_epoch_end(self, epoch, logs=None):
        print('For epoch {} loss is {:7.2f}.'.format(epoch, logs['loss']))


def artificial(data):
    print("Data is of shape {}, {}".format(data.shape[0], data.shape[1]))
    signals = np.reshape(data, (data.shape[0], data.shape[1], 1))
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, activation='relu', input_shape=(data.shape[1], 1)))
    lstm_model.add(RepeatVector(data.shape[1]))
    lstm_model.add(LSTM(100, activation='relu', return_sequences=True))
    lstm_model.add(TimeDistributed(Dense(1)))

    lstm_model.compile(optimizer='adam', loss='mse')

    print("I'm about to train")
    hist = lstm_model.fit(signals, signals, epochs=100, verbose=0,
                          callbacks=[LossAndErrorPrintingCallback()])


    lstm_model = Model(inputs=lstm_model.inputs, outputs=lstm_model.layers[1].output)

    representations = lstm_model.predict(signals)


def get_representations(signals, default_length, latent_dim):
    """
    Function builds and trains an autodecoder for representation learning
    :param signals: numpy array of signals of dimensions Nxdefault_length
    :return: numpy array of representations #TODO
    """
    signals = np.reshape(signals, (signals.shape[0], default_length, 1))
    print ("Trying to get signal representations for signals of shape ({}, {}, {})".format(signals.shape[0], signals.shape[1], signals.shape[2]))
    lstm_model = Sequential()
    lstm_model.add(LSTM(latent_dim, activation='relu', input_shape=(default_length, 1)))
    lstm_model.add(RepeatVector(default_length))
    lstm_model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    lstm_model.add(TimeDistributed(Dense(1)))

    lstm_model.compile(optimizer='adam', loss='mse')

    print("I'm about to train")
    hist = lstm_model.fit(signals, signals, batch_size=50, epochs=1, verbose=0, callbacks=[LossAndErrorPrintingCallback()])

    print("Done training, this the history")
    print(hist.history.keys())

    lstm_model = Model(inputs=lstm_model.inputs, outputs=lstm_model.layers[1].output)

    representations = lstm_model.predict(signals)
    return representations

def pad_signal(signal, default_length):
    """
    Fills signal to default length by zero-padding
    :return: signal of length default_length with zeros added at the end of existing signal
    """
    return np.pad(signal, (0, default_length - len(signal)), 'constant')

def slice_and_pad(signal, default_length):
    """
    Slices long signals for training
    :param default_length: length of return signals
    :return: list of signals of default length
    """
    sliced_signals = []
    for i in range(0, len(signal), default_length):
        sliced_signals.append(signal[i:i+default_length])
    sliced_signals[-1] = pad_signal(sliced_signals[-1], default_length)
    return sliced_signals

def preprocess_signals(signals, default_length):
    """
    Preprocesses signals by slicing long ones to default length and zero-padding the short ones
    :param signals: numpy array of signals
    :return: numpy array of preprocessed signals
    """
    conservative_signals = []
    for file in signals:
        for signal in file:
            if len(signal) <= default_length:
                conservative_signals.append(pad_signal(signal, default_length))
            else:
                conservative_signals.extend(slice_and_pad(signal, default_length))
    return np.asarray(conservative_signals)

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
    return len(lens), mean, std

def plot_signals(signals):
    for file in signals:
        for signal in file:
            plt.plot(signal)
            plt.show()


def find_most_common_length(lengths):
    values, counts = np.unique(lengths, return_counts=True)
    for value, count in zip(values, counts):
        if count > len(lengths)/10:
            print("Length {} occurs {} times".format(value, count))

if __name__ == "__main__":
    path = sys.argv[1]
    default_len = 10000
    latent_dim = 100
    all_signals = parse(path)
    #plot_signals(all_signals)
    num_of_signals, len_mean, len_std = len_statistics(all_signals)
    preprocessed_signals = preprocess_signals(all_signals, default_len)
    print("The average length of {} signals is {} with standard deviation of {}".format(num_of_signals, len_mean, len_std))
    get_representations(signals=preprocessed_signals, default_length=default_len, latent_dim=latent_dim)



