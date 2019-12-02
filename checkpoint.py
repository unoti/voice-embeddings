# Save checkpoints of models periodically

import os
import os.path
import time

class CheckpointMonitor:
    """Periodically saves models, and manages rolling checkpoint files."""
    def __init__(self, model, directory, base_name, seconds_between_saves=120, log_fn=None):
        self.model = model
        self.directory = directory
        self.base_name = base_name
        self.seconds_between_saves = seconds_between_saves
        self.csv_filename = os.path.join(directory, base_name + '.csv')
        self._log_fn = log_fn
        self.batch_num = 0 # Might be updated in self._init_csv()

        self._next_checkpoint_time = 0 # Will be set below.
        self._csv_file = self._init_csv()
        self._csv_updates = [] # We'll write these to the csv file when we save the model.
        self._reset_checkpoint_time()
    
    def train_step_done(self, training_loss, test_loss=None):
        """Call this periodically during the training process.
        The CheckpointMonitor will save the model if appropriate.
        Returns true if the model was saved, or false if we determined
        now isn't an appropriate time to save.
        """
        self.batch_num += 1
        self._append_csv(training_loss, test_loss)
        if self.is_save_needed():
            self.save()
            return True
        else:
            return False
    
    def save(self):
        """Definitely save the model without regard for whether we've
        saved recently.  You can call this at the end of your training
        to make sure the latest model is saved.
        """
        filename = self._make_filename()
        self._log('Saving', filename)
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        self.model.save_weights(filename)

        for line in self._csv_updates:
            self._csv_file.write(line + '\n')
        self._csv_updates = []            
        self._csv_file.flush()

        self._reset_checkpoint_time()
        self._log('Saved', filename)
    
    def load_most_recent(self):
        """Load the most recent model.  Returns the model loaded, or None if no model found.
        """
        # TODO: find most recent.
        # TODO: keep most recent N files.  Maybe keep several of different ages (one week, one day, one hour)
        filename = self._make_filename()
        loaded = False
        if os.path.exists(filename):
            self._log('Loading model from', filename)
            # Don't use model.load_model() because we use lambda layers in our model. Use load_weights instead.
            # See https://github.com/keras-team/keras/issues/5298
            self.model.load_weights(filename)
            self._log('Preloaded model from', filename)
            loaded = True
        return loaded

    def is_save_needed(self):
        return time.time() >= self._next_checkpoint_time
    
    def _reset_checkpoint_time(self):
        self._next_checkpoint_time = time.time() + self.seconds_between_saves

    def _make_filename(self):
        filename = os.path.join(self.directory, self.base_name) + '.h5'
        return filename
    
    def _log(self, *params):
        if not self._log_fn:
            return
        self._log_fn(*params)
    
    def _init_csv(self):
        if os.path.exists(self.csv_filename):
            last_line = read_last_line(self.csv_filename)
            self._csv_file = open(self.csv_filename, 'a')
            parts = last_line.split(',')
            try:
                last_batch = int(parts[0])
            except ValueError:
                last_batch = 0
                self._log('Could not extract last batch number from csv file. Using 0.')
            self.batch_num = last_batch
            self._log('Resuming at batch', self.batch_num)
        else:
            self._csv_file = open(self.csv_filename, 'w')
            self._csv_file.write('batch_num,loss,test_loss\n')
        return self._csv_file
    
    def _append_csv(self, training_loss, test_loss):
        if test_loss==None:
            test_loss=''
        parts = [self.batch_num, training_loss, test_loss]
        line = ','.join([str(n) for n in parts])
        self._csv_updates.append(line)
        # We will update the csv file next time we write the model.
        # This is so that the information in the csv file stays consistent with the last saved model.

def read_last_line(filename):
    # https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file/3346788
    # max_line_len = 1024
    # with open(filename, 'rb') as f:
    #     first = f.readline()
    #     f.seek(-2, os.SEEK_END) # Jump to the second last byte.
    #     while f.tell() > 1 and f.read(1) != b'\n': # Is this byte an eol?
    #         f.seek(-2, os.SEEK_CUR) # Jump back 2 bytes (because previous line went forward 1).
    #     f.seek(-1, os.SEEK_CUR) # This is probably not needed on unix-like line ending systems
    #     last = f.readline().decode()
    # return last
    # I struggled so much with the above stuff that I just decided to do this the naive way:
    with open(filename, 'r') as f:
        last_line = None
        for line in f:
            last_line = line
        last_line = last_line.strip()
        return last_line