# Factory methods for producing things the application needs.
# Generally the strategy is for this module to read the configuration
# and configure things as appropriate for the application and produce
# the configured objects needed for the application to run.
# So we should move any references to config in to here.
# This isn't completely done yet, so that's aspirational.

import config
from checkpoint import CheckpointMonitor
import minibatch
import models
from speakerdb import SpeakerDatabase

import keras.models
import os
from tensorflow.python.util import deprecation
import tensorflow as tf

# Singletons
_speaker_db = None

def make_speaker_db() -> SpeakerDatabase:
    global _speaker_db
    if not _speaker_db:
        _speaker_db = SpeakerDatabase(config.DATASET_TRAINING_DIR)
    return _speaker_db

def make_model() -> keras.models.Model:
    """Returns an untrained model."""
    model = models.make_model(config.BATCH_SIZE, config.EMBEDDING_LENGTH,
        config.NUM_FRAMES, config.NUM_FILTERS, config.LEARNING_RATE)
    return model

def load_model() -> keras.models.Model:
    model = make_model()
    checkpoint_monitor = make_checkpoint_monitor(model)
    checkpoint_monitor.load_most_recent()
    return model

def make_checkpoint_monitor(model: keras.models.Model) -> CheckpointMonitor:
    return CheckpointMonitor(model, directory=config.CHECKPOINT_DIRECTORY, base_name='voice-embeddings',
        seconds_between_saves=config.CHECKPOINT_SECONDS, log_fn=log)

def make_batch() -> minibatch.MiniBatch:
    make_speaker_db()
    return minibatch.create_batch(_speaker_db)

def log(*items):
    print(*items)

def init():
    """Call this once at application startup."""
    # I hate to suppress warnings, but there are so many layers of software printing so many warnings that
    # I struggle to even make my application code work, because it gets swamped in all this bullshit.
    # Make it stop!
    # Layers that are fucking spamming me include: Intel KMP, keras, and tensorflow.

    # Stop console spam. https://stackoverflow.com/questions/56224689/reduce-console-verbosity
    os.environ['KMP_WARNINGS'] = 'off'
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)