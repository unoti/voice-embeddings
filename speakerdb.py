import os
import random

class SpeakerDatabase:
    """A dataset of audio files from various speakers.
    The directory structure of the data is one directory per speaker, with the speaker id as the name
    of the directory.
    Under the speaker directory multiple directories, each containing audio clips from that speaker.
    """
    def __init__(self, directory):
        """
        directory: path name of the root directory.
        This directory should contain directories named by the speaker ids.
        """
        self.directory = directory
        self.speaker_ids = os.listdir(self.directory)
        if not self.speaker_ids:
            raise Exception('Speaker database at %s should contain folders named by speaker id')
    
    def random_speaker(self):
        """Returns the id of a random speaker."""
        return random.choice(self.speaker_ids)
    
    def random_wav(self):
        """Selects a random speaker, then returns a random utterance from that speaker.
        returns speaker_id, wav_filename.
        """
        speaker_id = self.random_speaker()
        wav_filename = self.random_wav_for_speaker(speaker_id)
        return speaker_id, wav_filename

    def random_wav_for_speaker(self, speaker_id):
        speaker_root = os.path.join(self.directory, speaker_id)
        speaker_files = []
        for root, dirs, filenames in os.walk(speaker_root):
            for partial_fnam in filenames:
                filename = os.path.join(root, partial_fnam)
                speaker_files.append(filename)
        return random.choice(speaker_files)
    
    def random_triplet(self):
        """Selects a triplet of audio samples. Two from the same speaker, and one from a different speaker.
        Returns filenames: anchor, positive, negative.
        anchor: A wav filename with an audio sample for the "anchor" speaker, which will match the positive example.
        positive: A wave filename with an audio sample for the "positive" speaker, which is from the same speaker as the anchor.
        negative: A wave filename with an audio sample from a different speaker from the anchor.
        """
        anchor_id, anchor_wav = self.random_wav()
        while True:
            positive_wav = self.random_wav_for_speaker(anchor_id)
            if positive_wav != anchor_wav:
                break
        while True:
            negative_id, negative_wav = self.random_wav()
            if negative_id != anchor_id:
                break
        return anchor_wav, positive_wav, negative_wav