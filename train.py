import minibatch
import application
from application import log

import keras.models
import numpy as np

def _train_batch(model: keras.models.Model, batch: minibatch.MiniBatch):
    X, _ = batch.inputs()
    # Y isn't used because it's the speaker ids as strings.
    # We're not actually trying to predict Y in this model, we're just optimizing the embeddings.
    # So just generate numbers for Y.  I expect zeros or ones would work just as well, since Y
    # isn't involved in the loss function.  Is this true?
    Y = np.random.uniform(size=(X.shape[0], 1))
    loss = model.train_on_batch(X, Y)
    return loss

def train():
    application.init()

    log('Building model')
    #*TODO: Detect if the pre-saved model doesn't exist.
    model = application.make_model()
    checkpoint_monitor = application.make_checkpoint_monitor(model)
    checkpoint_monitor.load_most_recent()
    batch_num = checkpoint_monitor.batch_num # Restored from the csv file.
    #log('Model learning rate set to %f' % model.lr.get_value())
    
    log('Building speaker db')
    speaker_db = application.make_speaker_db()

    while True:
        batch_num += 1
        log('Building batch {0}'.format(batch_num))
        batch = minibatch.create_batch(speaker_db)

        log('Training')       
        loss = _train_batch(model, batch)
        log('batch {0} loss={1}'.format(batch_num, loss))

        #*TODO: log test_loss
        #if checkpoint_monitor.is_save_needed():
        #    test_loss = ...
        checkpoint_monitor.train_step_done(loss, test_loss=None)


if __name__ == '__main__':
    train()