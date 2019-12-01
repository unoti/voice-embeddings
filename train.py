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
    model = application.make_model()
    
    log('Building speaker db')
    speaker_db = application.make_speaker_db()

    batch_num = 0
    while True:
        batch_num += 1
        log('building batch {0}'.format(batch_num))
        batch = minibatch.create_batch(speaker_db)
        log('training')
        loss = _train_batch(model, batch)
        log('batch {0} loss={1}'.format(batch_num, loss))

if __name__ == '__main__':
    train()