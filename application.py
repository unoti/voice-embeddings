# Factory methods for producing things the application needs.

import config
from speakerdb import SpeakerDatabase

def make_speaker_db() -> SpeakerDatabase:
    return SpeakerDatabase(config.DATASET_TRAINING_DIR)