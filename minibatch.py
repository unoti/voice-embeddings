from speakerdb import SpeakerDatabase
import audiolib
import config

import numpy as np

class MiniBatch:
    """
    A minibatch is a group of training triplets in matrix form assembled from audio samples.
    """
    def __init__(self, samples, speaker_ids):
        """
        The samples will be sequenced as all of the anchors first, then all of the positives, then
        all of the negatives.
        samples: A list of audio samples. Each sample is of shape (num_frames, num_filters, 1).
        speaker_ids: The speaker id for each sample in **samples**.
        """
        self.X = np.array(samples)
        self.Y = np.array(speaker_ids)

    def inputs(self):
        """
        Returns (X, Y)
        X: A tensor of shape (batch_size * 3, num_frames=160, num_filter_banks=64, 1)
        Y: The speaker id of each sample in X.
        """
        return self.X, self.Y

def _clipped_audio(x, num_frames):
    """
    Truncate an audio clip to be at most num_frames.
    If the input is longer than num_frames then select a random subsection.
    """
    if x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x

def _get_audio_features(filename):
    """Load audio from a file and process it into filter bank frequencies and time slices."""    
    rate, data = audiolib.load_wav(filename)
    features = audiolib.extract_features(data, sample_rate=rate, num_filters=config.NUM_FILTERS)
    return features

def _add_samples(batch, samples):
    """Add samples into a batch, truncating each sample as needed."""
    for sample in samples:
        batch.append(_clipped_audio(sample, config.NUM_FRAMES))

def create_batch(db: SpeakerDatabase) -> MiniBatch:
    # Select and preprocess all of the triplets we'll use in this batch.
    anchors = []
    positives = []
    negatives = []
    anchor_ids = []
    negative_ids = []
    for _ in range(config.BATCH_SIZE):
        anchor_fnam, positive_fnam, negative_fnam, anchor_id, negative_id = db.random_triplet()
        anchors.append(_get_audio_features(anchor_fnam))
        positives.append(_get_audio_features(positive_fnam))
        negatives.append(_get_audio_features(negative_fnam))
        anchor_ids.append(anchor_id)
        negative_ids.append(negative_id)
    
    # Assemble the batch.  Sequencing is all of the anchors, then the positives, then the negatives.
    batch = []
    _add_samples(batch, anchors)
    _add_samples(batch, positives)
    _add_samples(batch, negatives)

    # Assemble the speaker ids of the samples in the batch in the same sequence.
    batch_ids = []
    batch_ids.extend(anchor_ids) # Anchor ids.
    batch_ids.extend(anchor_ids) # Positive ids, which are the same as the anchor ids.
    batch_ids.extend(negative_ids) # Negative ids
    return MiniBatch(batch, batch_ids)