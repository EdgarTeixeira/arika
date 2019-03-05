import math

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import SGD


class SnapshotEnsemble(Callback):
    def __init__(self,
                 iterations_per_epoch,
                 epochs,
                 cycles,
                 initial_lr=0.02,
                 final_lr=0.001):
        if cycles > epochs:
            raise ValueError("Cycles must be less than or equal to epochs")
        if initial_lr < final_lr:
            raise ValueError(
                "Initial learning rate must be greater than the final one.")

        self.iterations_per_epoch = iterations_per_epoch
        self.epochs = epochs
        self.cycles = cycles
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self._snapshots = []

    @property
    def snapshots(self):
        ensemble = []
        for s in self._snapshots:
            model = clone_model(self.model)
            model.set_weights(s)
            ensemble.append(model)

        return ensemble

    def schedule(self, it):
        cycle_size = math.ceil(
            self.iterations_per_epoch * self.epochs / self.cycles)

        lr = (math.pi * (it % cycle_size)) / cycle_size
        lr = (math.cos(lr) + 1) / 2
        lr = (self.initial_lr - self.final_lr) * lr + self.final_lr

        return lr

    def on_train_begin(self, logs=None):
        if not isinstance(self.model.optimizer, SGD):
            raise ValueError(
                "You have to use the SGD optimizer with Snapshot Ensembles")

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        # epoch starts at zero
        checkpoints = math.ceil(self.epochs / self.cycles)
        if epoch % checkpoints == 0:
            self._snapshots.append(self.model.get_weights())

    def on_batch_begin(self, batch, logs=None):
        # batch stats at zero
        time = batch + (self._current_epoch * self.iterations_per_epoch)
        learning_rate = self.schedule(time)
        K.set_value(self.model.optimizer.lr, learning_rate)
