from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import random

def load_wav(wav_file):
    """Load a wav file.
    Returns:
    rate: number of samples per second.
    data: an array of samples as signed 16-bit integers.
    """
    rate, data = wavfile.read(wav_file)
    return rate, data

def graph_spectrogram(wav_file):
    """Plots a spectrogram for a wav audio file."""
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def graph_audio(wav_file):
    """Plots 2 graphs for an audio file: an amplitude graph, and a spectrogram."""
    rate, samples = get_wav_info(wav_file)
    plt.figure(1)    
    a = plt.subplot(211)    
    a.set_xlabel('time [s]')    
    a.set_ylabel('value [-]')    
    plt.plot(samples)
    c = plt.subplot(212)
    Pxx, freqs, bins, im = c.specgram(samples, NFFT=1024, Fs=16000, noverlap=900)    
    c.set_xlabel('Time')    
    c.set_ylabel('Frequency')    
    plt.show()

