import config

import keras.backend as K

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    # https://keras.io/backend/#squeeze
    # https://keras.io/backend/#batch_dot
    # Calculate the dot product for the entire batch, where the input batch
    # is shaped like (batch_size, :).  Result size has fewer dimensions than the
    # input but is expanded to have at least 2 dimensions.  Then we squeeze that
    # removing one dimension.
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    return dot

def deep_speaker_loss(y_true, y_pred):
    # y_true.shape = (batch_size, embedding_size)
    # y_pred.shape = (batch_size, embedding_size)
    # CONVENTION: Input is:
    # concat(BATCH_SIZE * [ANCHOR, POSITIVE_EX, NEGATIVE_EX] * NUM_FRAMES)
    # EXAMPLE:
    # BATCH_NUM_TRIPLETS = 3, NUM_FRAMES = 2
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # ANCHOR 3 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # POS EX 3 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # NEG EX 3 (512,)
    # _____________________________________________________

    #elements = int(y_pred.shape.as_list()[0] / 3)
    elements = config.BATCH_SIZE

    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]

    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + config.ALPHA, 0.0)
    total_loss = K.sum(loss)
    return total_loss
