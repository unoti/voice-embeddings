from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from python_speech_features import fbank

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
    rate, samples = load_wav(wav_file)
    graph_raw_audio(samples)

def graph_raw_audio(samples):
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

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]

def extract_features(signal=np.random.uniform(size=48000), sample_rate=16000, num_filters=64):
    """
    Returns: np.array of shape (num_frames, num_filters, 1). Each frame is 25 ms.
    """
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=num_filters, winlen=0.025)   #filter_bank (num_frames , 64),energies (num_frames ,)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    #delta_1 = normalize_frames(delta_1)
    #delta_2 = normalize_frames(delta_2)

    #frames_features = np.hstack([filter_banks, delta_1, delta_2])    # (num_frames , 192)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)
    return np.reshape(np.array(frames_features),(num_frames, num_filters, 1))   #(num_frames,64, 1)
