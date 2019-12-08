from collections import deque
import pyaudio
import time
import numpy as np

def record(seconds, rate=16000):
    """Record a few seconds of audio.
    Normally AudioStream is used for continuous processing of an ongoing audio stream,
    but you can use this method to record a few seconds of audio and stop.
    Returns a numpy array of 16-bit signed integers.
    """
    s = AudioStream(seconds=seconds, rate=rate)
    s.start()
    print('Recording')
    for i in range(seconds):
        print('%d seconds remaining' % (seconds - i))
        time.sleep(1)
    s.stop()
    print('Recording finished')
    return s.sound_array()

# This implementation uses a deque.  It might be more efficient to use io.BytesIO
# like this does: https://github.com/hbock/byte-fifo/blob/master/fifo.py
# What I've written here does some unknown amount of byte copying.
# I'm going to just use what I've written here for now and optimize later if needed.
class AudioStream:
    """
    Streams audio recording in real time, providing raw sound data from the last few seconds.
    """
    def __init__(self, seconds=4, rate=44100, bytes_per_sample=2):
        self.buffer_seconds = seconds
        self.rate = rate
        self.bytes_per_sample = bytes_per_sample
        buffer_size = seconds * rate * bytes_per_sample
        self._buffer = deque(maxlen=buffer_size)
        self._pyaudio = None # pyaudio.PyAudio object will be initialized in self.start().
        self._stream = None # Stream object will be initialized in self.start().
        
        seconds_per_buffer = 0.2 # How much audio we want included in each callback.
        self._frames_per_buffer = int(rate * seconds_per_buffer) # How many samples we get per callback.
        self._stop_requested = False
    
    def start(self):
        """Start recording the audio stream."""
        self._pyaudio = pyaudio.PyAudio() # (1) Instantiate PyAdio. Sets up the portaudio system.
        pyaudio_format = self._pyaudio.get_format_from_width(width=2)
        self._stream = self._pyaudio.open(format = pyaudio_format,
                        channels = 1,
                        rate = self.rate,
                        input = True,
                        frames_per_buffer = self._frames_per_buffer,
                        stream_callback = self._pyaudio_callback)
        self._stream.start_stream()

    def stop(self):
        """Stop recording the audio stream."""
        self._stop_requested = True
        while self._stream.is_active():
            time.sleep(0.1)
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()
    
    def sound_data(self):
        """Returns an iterable sequence of bytes in the buffer representing the most recent buffered audio."""
        return self._buffer
    
    def sound_array(self):
        """Returns a numpy array with 16-bit signed integers representing
        samples of the most recent buffered audio.
        """
        b = bytearray(self._buffer)
        return np.frombuffer(b, dtype=np.int16)

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """Called by pyaudio to process a chunk of data.
        This is called in a separate thread.
        """
        self._buffer.extend(in_data)
        if self._stop_requested:
            flag = pyaudio.paComplete
        else:
            flag = pyaudio.paContinue
        return (None, flag)