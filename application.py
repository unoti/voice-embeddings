# Factory methods for producing things the application needs.
# Generally the strategy is for this module to read the configuration
# and configure things as appropriate for the application and produce
# the configured objects needed for the application to run.
# So we should move any references to config in to here.
# This isn't completely done yet, so that's aspirational.

import config
from speakerdb import SpeakerDatabase
import models
import keras.models

import os

def make_speaker_db() -> SpeakerDatabase:
    return SpeakerDatabase(config.DATASET_TRAINING_DIR)

def make_model() -> keras.models.Model:
    return models.make_model(config.BATCH_SIZE, config.EMBEDDING_LENGTH, config.NUM_FRAMES, config.NUM_FILTERS)

def log(*items):
    print(*items)

def init():
    """Call this once at application startup."""
    # Stop console spam. https://stackoverflow.com/questions/56224689/reduce-console-verbosity
    os.environ['KMP_WARNINGS'] = 'off'
