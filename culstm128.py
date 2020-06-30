import pretty_midi as pm
import numpy as np
import os
from tensorflow.keras import utils
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
# tf.config.gpu_options.allow_growth = True

def train_network():
    """ Train a Neural Network to generate music """
    network_output, network_input = get_notes()

    # get amount of pitch names
    # n_vocab = len(set(notes))
    n_entries = network_input.shape[2]
    network_input = np.transpose(network_input, axes = [2,0,1])
    print(network_input.shape)

    model = create_network(network_input)

    train(model, network_input, network_output)

def get_notes():
    x_data, y_data, artists = read_folder('data1/')
    x_data_final_t, y_data_final_t, artistlistfinal = transpose(x_data, y_data, artists)
    print(x_data)
    print(x_data.shape)
    print(x_data_final_t)
    print(x_data_final_t.shape)

    return artistlistfinal, x_data_final_t



def create_network(network_input):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        128,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))

    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = 'accuracy')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-signe-disc-128-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, validation_split = 0.2, epochs=500, batch_size=96, callbacks=callbacks_list)
def transpose(x_data_final, y_data_final, artistlist):
    x_data_final_t, y_data_final_t = np.empty((96, 4, 0)), np.empty((96, 4, 0))
    artistlistfinal = []
    for i in range(12):
        x_transposed = x_data_final + i - 6
        x_data_final_t = np.concatenate((x_transposed, x_data_final_t), axis=2)
        x_data_final_t[x_data_final_t < 10] = 0
        y_transposed = y_data_final + i - 6
        y_data_final_t = np.concatenate((y_transposed, y_data_final_t), axis=2)
        y_data_final_t[x_data_final_t < 10] = 0
    for artist in artistlist:
        for i in range(12):
            artistlistfinal.append(artist)
    artistlistfinal = [artist == 'signe' for artist in artistlistfinal]

    artistlistfinal = np.array(artistlistfinal)
    artistlistfinal = 1*artistlistfinal
    print(artistlistfinal)
    print(artistlistfinal.shape)
    


    return x_data_final_t, y_data_final_t, artistlistfinal


def read_folder(directory_in_str):
    directory = os.fsencode(directory_in_str)
    artistlist = []

    x_data_final, y_data_final = np.empty((96, 4, 0)), np.empty((96, 4, 0))
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mid"):
            artist = filename.split('_')[0]
            midi_data = pm.PrettyMIDI(directory_in_str + filename)
            piano_roll = midi_info(midi_data)
            x, y, artists = piece_segments(piano_roll, artist)
            x_data_final = np.concatenate((x, x_data_final), axis=2)
            y_data_final = np.concatenate((y, y_data_final), axis=2)
            for artist in artists:
                artistlist.append(artist)
            continue
        else:
            continue
    # print(artistlist)
    # artistlist = [artist == 'signe' for artist in artistlist]

    # artistlist = np.array(artistlist)
    # artistlist = utils.to_categorical(artistlist)
    # print(artistlist)
    # print(artistlist.shape)



    return x_data_final, y_data_final, artistlist


def midi_info(midi_file):
    tempo = (midi_file.get_tempo_changes()[-1])
    midi_mono = midi_file.instruments[0]

    # bpm to seconds, then divided by beats in a bar, multiplied by 96 ticks
    fs = int(tempo) / 60 * 96 / 4
    piano_roll = midi_mono.get_piano_roll(fs=fs)

    return piano_roll


def piece_segments(piano_roll, artist):
    """
    chooses the segments that are processed as x and y
    Uses Data Segment
    """
    x_piece_data, y_piece_data = np.empty((96, 4, 0)), np.empty((96, 4, 0))
    artists = []

    for i in range(piano_roll.shape[1] // 768):  # 96 ticks, * 4 bars * call and response
        x_start = i * 768
        x_end = x_start + 384
        piano_roll_segment_x = piano_roll[:, x_start: x_end]
        piano_roll_segment_y = piano_roll[:, x_end + 1: x_end + 1 + 384]
        x_piece_data = np.concatenate((data_segment(piano_roll_segment_x), x_piece_data), axis=2)
        y_piece_data = np.concatenate((data_segment(piano_roll_segment_y), y_piece_data), axis=2)
        artists.append(artist)


    return x_piece_data, y_piece_data, artists


def data_segment(piano_roll_segment):
    """
    :param piano_roll_segment: section of piano roll to convert to matrix
    :return: the summed segment
    """
    new_data = []
    index = -1
    for i in piano_roll_segment:
        index += 1
        new_values =[]
        for values in i:
            if values > 0:
                new_values.append(index)
            else:
                new_values.append(0)
        new_data.append(new_values)
    np_data = np.array(new_data)
    segment = np_data.sum(axis=0).reshape(96,4,1)
    return segment

if __name__ == '__main__':
    train_network()
