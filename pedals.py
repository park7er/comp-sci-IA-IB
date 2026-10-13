
import numpy as np
import soundfile as sf
import pedalboard as pb
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os
print(os.getcwd())

file_path = r"C:\Users\sopto\OneDrive\桌面\code\ia\pedals1\input_sample.wav"

#This is how you import an audio file.
audio, samplerate = sf.read(file_path) 

#samplerate = 44100.0


def tube_screamer(audio, volume=10, drive=10, tone=10):

    board = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz = 1100),
                           pb.HighpassFilter(cutoff_frequency_hz = 150),
                           pb.Gain(gain_db = 18 * drive / 10),
                           pb.Compressor(),
                           pb.LowpassFilter(cutoff_frequency_hz = 800 + 2000*tone/10),
                           pb.Gain(gain_db = volume / 10),
                           pb.Clipping()])

    return board.process(audio, samplerate)

def big_muff(audio, volume = 10, sustain = 10, tone = 10):
    
    board = pb.Pedalboard([pb.Distortion(drive_db = 25 * sustain / 10),
                           pb.LowShelfFilter(cutoff_frequency_hz = 800,
                                             gain_db = 4 * (10 - tone)/10 + 1),
                           pb.HighShelfFilter(cutoff_frequency_hz = 1200,
                                              gain_db = 4 * tone / 10 + 1),
                           pb.Gain(gain_db = volume / 10),
                           pb.Clipping()])

    return board.process(audio, samplerate)
    


def boss_ds1(audio, volume = 10, distortion = 10, tone = 10):
    
    board = pb.Pedalboard([pb.Distortion(drive_db = 18 * distortion / 10),
                           pb.LowShelfFilter(cutoff_frequency_hz = 1000,
                                             gain_db = 4 * (10 - tone)/10 + 1),
                           pb.HighShelfFilter(cutoff_frequency_hz = 1100,
                                              gain_db = 4 * tone / 10 + 1),
                           pb.Gain(gain_db = volume / 10),
                           pb.Clipping()])

    return board.process(audio, samplerate)
    


def proco_rat(audio, volume = 10, distortion = 10, rat_filter = 10):

    tone = 10 - rat_filter

    board = pb.Pedalboard([pb.HighpassFilter(cutoff_frequency_hz = 40.0),
                           pb.Gain(gain_db = 18 * distortion / 10.0),
                           pb.Clipping(),
                           pb.Distortion(drive_db = 12.0),
                           pb.LowpassFilter(cutoff_frequency_hz = 800 + 3000 * tone / 10.0),
                           pb.Gain(gain_db = volume / 10.0),
                           pb.Clipping()])
    return board.process(audio, samplerate)



def generate_white_noise(duration, sr=44100):
    return np.random.normal(0, 1, int(sr * duration))


# generate_white_noise("./20_seconds_white_noise.wav")

#effected = proco_rat(audio)


# Examples of all four pedals with their knobs at max settings.
# sf.write("./rat_max_output_sample.wav", proco_rat(audio), samplerate)
# sf.write("./boss_ds1__max_output_sample.wav", boss_ds1(audio), samplerate)
# sf.write("./big_muff__max_output_sample.wav", big_muff(audio), samplerate)
# sf.write("./tube_screamer__max_output_sample.wav", tube_screamer(audio), samplerate)


# # Examples of two of the four pedals with their knobs set at noon.
# sf.write("./rat_knobs_noon_output_sample.wav", proco_rat(audio, volume = 5, distortion = 5, rat_filter = 5), samplerate)
# sf.write("./boss_ds1_knobs_noon_output_sample.wav", boss_ds1(audio, volume = 5, distortion = 5, tone = 5), samplerate)


def sin_wave(file_location, length_in_seconds=20, frequency=440, sampling_rate=44100, amplitude=0.5):
    # Generate time array
    time = np.linspace(0, length_in_seconds, int(sampling_rate * length_in_seconds), endpoint=False)

    # Generate sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)

    # Save sine wave as a WAV file
    sf.write(file_location, sine_wave, sampling_rate)
    print('sin wave generated')


#sin_wave('sine_wave.wav', length_in_seconds=20)

