
# Things you definitely should change

# Dataset format: one directory for each speaker where the speaker id is the name of the directory.
# Under that is many folders containing wav files for that speaker.
DATASET_TRAINING_DIR = r'd:\datasets\voxceleb1\vox1\wav' # Training dataset.

# ----------
# Things you could safely change
CHECKPOINT_SECONDS = 600 # Save the model after processing every n seconds.
CHECKPOINT_DIRECTORY = 'checkpoints' # Directory to save models during training.

# ----------
# Things you probably shouldn't change

BATCH_SIZE = 32 # Must be even.
# Alpha, as used in FaceNet https://arxiv.org/pdf/1503.03832.pdf .
# Alpha is how close the embeddings of two samples need to be to be considered the same person.
ALPHA = 0.2 # as used in FaceNet https://arxiv.org/pdf/1503.03832.pdf .
NUM_FRAMES = 160 # Each frame is advanced by 10ms, so so 160 frames is 1.6 seconds
EMBEDDING_LENGTH = 512 # How many features are in a speaker embedding.
NUM_FILTERS = 64 # Number of FFT frequency filter bands used to create embeddings.
