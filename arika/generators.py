import math

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SparseSequence(Sequence):
    """
    Sequence for dealing with a sparse dataset in keras. This class assumes
    that only the x matrix is sparse.
    """

    def __init__(self, sparse_x, y, batch_size=32, shuffle=True):
        """
        Parameters
        ----------
        sparse_x: sparse_matrix
            The features matrix. It assumes that this class has a
            todense method that converts the sparse representation
            into a dense one
        y: dense matrix or vector
            The labels of the problem. It assumes they are in a
            dense format.
        batch_size: int, default=32
            The size of the batches. The last batch may be smaller
        shuffle: bool, deafault=True
            A flag to control if the dataset should be shuffled
            after each epoch.
        """
        self.x = sparse_x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]\
                      .todense()
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            idx = np.arange(self.y.shape[0])
            np.random.shuffle(idx)

            self.x = self.x[idx]
            self.y = self.y[idx]


class VariableLengthTextSequence(Sequence):
    """
    Sequence for dealing with variable-length text in keras. Each sequence in
    a batch will be padded to have the same size as the longest sequence in
    that batch.
    """

    def __init__(self, x, y, batch_size=32, pad_value=0):
        """
        Paramters
        ---------
        x: list of sequences
            The features of the problem
        y: matrix or vector
            The labels of the problem
        batch_size: int, default=32
            The size of the batches. The last batch may be smaller
        pad_value: int, default=0
            The value to pad the smaller sequences
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.pad_value = pad_value

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = pad_sequences(batch_x, value=self.pad_value)

        if isinstance(self.y, list):
            batch_y = [
                y[idx * self.batch_size:(idx + 1) * self.batch_size]
                for y in self.y
            ]
        else:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
