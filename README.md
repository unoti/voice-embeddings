# voice-embeddings
Audio processing using deep neural networks. Speaker identification using voice embeddings.


## References
 * [Deep Speaker Paper](https://arxiv.org/pdf/1705.02304.pdf)
 * [Vox Celeb Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
 * [philipperemy/deep-speaker](https://github.com/philipperemy/deep-speaker)
 * [Walleclipse/Deep_Speaker-speaker_recognition_system](https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system)
 * [python_speech_features](https://github.com/jameslyons/python_speech_features)
    * ```python_speech_features.fbank()``` Produces Mel-filterbank energy features from an audio signal

## Structure of a Batch
* ```config.BATCH_SIZE=32```. A batch has 32 triplets in it.  Each triplet is composed of an anchor, positive, and negative.
* ```config.EMBEDDING_LENGTH=512```.  The embeddings are of length 512, so a speaker is characterized by 512 numbers.
* **Batch sequencing**. The total number of entries in a batch is ```config.BATCH_SIZE * 3``` because there are 3 samples in a triplet (anchor, positive, negative).  Each one of these entries is 512 numbers.  The sequence is as follows:
   * All the anchor samples
   * Then all the positive samples
   * Then all the negative samples

   | anchor | positive | negative |
   |--------|----------|----------|
   | 0-31   | 32-63    | 63-96    |

 * **The loss function**. The loss function defined in **triplet_loss**.py optimizes in both of these ways:
   * *maximize* the cosine similarity between the **anchor** examples and the **postive** examples
   * *minimize* the cosine similarity between the **anchor** examples and the **negative** examples

## Model
 * The shape of the input is ```(NUM_FRAMES, 64, 1)```.
 * ```config.NUM_FRAMES=160```.  Each frame is 25ms long, so by default this is 4 seconds of audio.
 * The shape of the output is ```config.EMBEDDING_LENGTH=512```.


## Next steps
**Get the model built**. Become very certain about what the inputs are to the model, and how the model trains.
Right now I'm a bit hazy on how the loss function works, and what the shapes of the inputs to the model are.
Looking at the loss function, it looks to me like it returns a single number.  But it looks to me like the model
contains many triplets.  So I need to gain crystal clarity on the following issues:
 * what are the shapes of the inputs in the model
 * what's the purpose of that resize with a 2048, and the /16 in there?
 * How does the loss function work with multiple samples? I mean, I understand how triplet loss works, but what's going on in their code?  Does that code I'm referring to even work? Consider comparing it to the original philipperemy code.
