from triplet_loss import deep_speaker_loss

import numpy as np
import math
import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda, Dense
from keras.models import Model

TRIPLETS_PER_BATCH = 3

def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)

def identity_block2(input_tensor, kernel_size, filters, stage, block):   # next step try full-pre activation
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_conv1_1')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_conv1.1_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.00001),
               name=conv_name_base + '_conv3')(x)
    x = BatchNormalization(name=conv_name_base + '_conv3_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_conv1_2')(x)
    x = BatchNormalization(name=conv_name_base + '_conv1.2_bn')(x)

    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

def conv_and_res_block(inp, filters, stage):
    conv_name = 'conv{}-s'.format(filters)
    o = Conv2D(filters,
                    kernel_size=5,
                    strides=2,
                    padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(l=0.00001), name=conv_name)(inp)
    o = BatchNormalization(name=conv_name + '_bn')(o)
    o = clipped_relu(o)
    for i in range(3):
        o = identity_block2(o, kernel_size=3, filters=filters, stage=stage, block=i)
    return o

def cnn_component(inp):
    x_ = conv_and_res_block(inp, 64, stage=1)
    x_ = conv_and_res_block(x_, 128, stage=2)
    x_ = conv_and_res_block(x_, 256, stage=3)
    #x_ = conv_and_res_block(x_, 512, stage=4) # This is the difference between the simple and complex model.
    return x_

def convolutional_model_simple(input_shape, batch_size, num_frames, embedding_length):
    """
    Builds a convolutional model that holds and entire batch of processed sound samples.
    input_shape: (NUM_FRAMES, NUM_FILTERS, 1)
    batch_size: (BATCH_SIZE * TRIPLETS_PER_BATCH)
    num_frames: Number of audio frames from config = 160 (=4sec because a frame is 25ms)
    embedding_length: number of features (floating point numbers) per output embedding.
    Returns: An uncompiled keras model.
    """
    # 
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.


    inputs = Input(shape=input_shape)
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/8, 64/8, 512)
    # -1 in the target size means that the number of dimensions in that axis will be inferred.
    # So this resize means: (n, num_frames/8, 2048)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 8), 2048)), name='reshape')(x)
    # Compute the average over all of the frames within a sample.
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  # .shape = (BATCH_SIZE, embedding_length)
    x = Dense(embedding_length, name='affine')(x)  # .shape = (BATCH_SIZE , embedding_length)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs, x, name='convolutional')
    return model

def make_model(batch_size, embedding_length, num_frames, num_filters):
    batch_size = batch_size * TRIPLETS_PER_BATCH
    # TODO: document exactly what's happening with the shape here.
    # [x] Become very certain what the shapes of x are coming out of batch.to_inputs() line 66 train.py
    # [x] make minibatch.X be of shape (batchsize, num_frames, 64, 1) <-- that's x shape. 
    #       then input_shape = (num_frames, 64, 1)
    # Shape of a single sample
    input_shape = (num_frames, num_filters, 1)
    model = convolutional_model_simple(input_shape, batch_size, num_frames, embedding_length)
    model.compile(optimizer='adam', loss=deep_speaker_loss)
    return model

def get_embedding(model, sample):
    """
    sample: A sample of shape (NUM_FRAMES, NUM_FILTERS, 1) which is (160,64,1).
        You can get this by using something like batch.X[n].
    model: a model.
    returns: an array of shape (NUM_EMBEDDING_LENGTH,) with the embeddings for this speaker.
    """
    input_batch = np.expand_dims(sample, axis=0) # .predict() wants a batch, not a single entry.
    emb_rows = model.predict(input_batch) # Predict for this batch of 1.
    embedding = np.squeeze(emb_rows) # Predict() returns a batch of results. Extract the single row.
    return embedding