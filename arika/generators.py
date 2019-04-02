import math

import numpy as np
from tensorflow import keras


class SparseSequence(keras.utils.Sequence):
    def __init__(self, sparse_x, y, batch_size=32, shuffle=True):
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


class VariableLengthSequence(keras.utils.Sequence):
    pass
