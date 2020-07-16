from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from music21 import converter, instrument, note, chord, stream, duration
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import tensorflow as tf
import pretty_midi
from fractions import Fraction
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def get_notes():
    """ Get all the notes and chords from the midi files """
    notes = []
    durations = []

    for file in glob.glob("data1/parker*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notesandRests
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, note.Rest):
                notes.append(' ')
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            durations.append(element.duration.type)


    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    with open('data/durations', 'wb') as filepath:
        pickle.dump(durations, filepath)


    return notes, durations

def prepare_sequences(notes, durations, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    durnames = sorted(set(item for item in durations))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    duration_to_int = dict((note, number) for number, note in enumerate(durnames))

    network_input = [[],[]]
    network_output = [[],[]]

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in_n = notes[i:i + sequence_length]
        sequence_out_n = notes[i + sequence_length]
        sequence_in_d = durations[i:i + sequence_length]
        sequence_out_d = durations[i + sequence_length]
        network_input[0].append([note_to_int[char] for char in sequence_in_n])
        network_input[1].append([duration_to_int[char] for char in sequence_in_d])
        network_output[0].append(note_to_int[sequence_out_n])
        network_output[1].append(duration_to_int[sequence_out_d])

    n_patterns = np.shape(network_input)[1]
    print(n_patterns)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 2))
    
    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
    network_output = utils.to_categorical(network_output)

    return (network_input, network_output)

def generate_notes(model, network_input, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)
    
    # Get pitch names and store in a dictionary
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    durnames = sorted(set(item for item in durations))
    int_to_dur = dict((number, duration) for number, duration in enumerate(durnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(100):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        print(prediction)

        index_n = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = numpy.append(pattern,index)
        #pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
  
def create_midi(prediction_output_n, prediction_output_d, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []
    print(prediction_output_n)
    print(prediction_output_d)
    # print(prediction_output)
    # print(np.shape(prediction_output))

    # create note and chord objects based on the values generated by the model
    for i in range(len(prediction_output_n)):
        item = prediction_output_n[i]
        pattern = item[0]
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif pattern == ' ':
            new_note = note.Rest()
            output_notes.append(new_note)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)



        # increase offset each iteration so that notes do not stack
        
        d=duration.Duration()
        d.type = prediction_output_d[i]
        offset += d.quarterLength/2
        print(offset)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 2)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss =[]
        
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates note sequences
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(LSTM(512, return_sequences = True))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)
      
    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load and convert the data
        notes, durations = get_notes()
        n_vocab = len(set(notes))
        X_train, y_train = prepare_sequences(notes, durations, n_vocab)

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training the model
        for epoch in range(epochs):

            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            #noise = np.random.choice(range(484), (batch_size, self.latent_dim))
            #noise = (noise-242)/242
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new note sequences
            gen_seqs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Training the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            # Print the progress and save into loss lists
            if epoch % sample_interval == 0:
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
              self.disc_loss.append(d_loss[0])
              self.gen_loss.append(g_loss)
        
        self.generate(notes, durations)
        self.plot_loss()
        
    def generate(self, input_notes, input_durations):
        # Get pitch names and store in a dictionary
        notes = input_notes
        durations = input_durations
        pitchnames = sorted(set(item for item in notes))
        durnames = sorted(set(item for item in durations))
        durnames = [d for d in durnames if d!='complex' and d!='zero' and d!='inexpressible']

        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        int_to_dur = dict((number, duration) for number, duration in enumerate(durnames))
        print(int_to_dur)
        print(int_to_note)
        # Use random noise to generate sequences
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        notes, durations = get_notes()
        n_vocab = len(pitchnames)
        d_vocab = len(durnames)
        predictions = np.reshape(predictions, (100,2))
        predictions = np.transpose(predictions)
        # print(predictions)


        pred_notes = [((x+1)*(n_vocab)/2) for x in predictions[0]]
        pred_durs = [((x+1)*(d_vocab)/2) for x in predictions[1]]
        pred_notes = [int_to_note[int(x)] for x in pred_notes]
        pred_durs = [int_to_dur[int(x)] for x in pred_durs]
        print(np.shape(pred_durs))
        print(np.shape(pred_notes))
        create_midi(pred_notes, pred_durs, 'gan_final_parker_10')
        
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

if __name__ == '__main__':
  gan = GAN(rows=100)    
  gan.train(epochs=10, batch_size=32, sample_interval=1)