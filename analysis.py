import numpy as np

def difference_emb(emb1, emb2):
    """Returns a scalar indicating the difference between 2 embeddings.
    Smaller numbers indicate closer.
    """
    dist = np.linalg.norm(emb1 - emb2)
    return dist

def difference_sample(model, sample1, sample2):
    """Returns a scalar indicating the difference between 2 sound samples.
    Smaller numbers indicate closer.
    """
    emb1 = models.get_embedding(model, sample1)
    emb2 = models.get_embedding(model, sample2)
    return difference_emb(emb1, emb2)

def comparison_matrix(model, batch, qty=3):
    """Compares the embeddings for the samples in a batch.
    All the embeddings in the batch will be placed along the rows and the columns,
    and the difference between each pair of embeddings will be placed in the matrix cells.
    """
    # It wouldn't surprise me if there exists a better, vectorized way to do this.
    embeddings = model.predict(batch.X)
    num_emb = len(embeddings)
    dists = np.zeros((qty, qty))
    qty = min(qty, num_emb)
    # Compare all the embeddings to each other
    for row in range(qty):
        row_emb = embeddings[row]
        for col in range(qty):
            col_emb = embeddings[col]
            dists[row][col] = difference_emb(row_emb, col_emb)
    return dists