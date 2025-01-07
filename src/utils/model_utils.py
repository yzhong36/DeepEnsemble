import numpy as np
import tensorflow as tf

def kmers_join(array, kmer = 3):

    pad_array = np.repeat("N", (kmer - 1)/2)
    pad_array = np.repeat(pad_array, array.shape[0]).reshape(array.shape[0], -1)

    array = np.append(pad_array, array, axis = 1)
    array = np.append(array, pad_array, axis = 1)

    array = np.array([[''.join(row[i:(i+kmer)]) for i in range(0,len(row)-kmer+1)] for row in array])

    return array

class BinaryCrossentropy_mask(tf.keras.losses.BinaryCrossentropy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, y_true_mask, y_pred):
        y_true, mask = tf.split(y_true_mask, 2, axis = -1)
        loss = tf.keras.losses.BinaryCrossentropy()(y_true * mask, y_pred * mask)

        return loss

class AUC_mask(tf.keras.metrics.AUC):
    def update_state(self, y_true_mask, y_pred, sample_weight=None):
        y_true, mask = tf.split(y_true_mask, 2, axis = -1)

        return super().update_state(y_true * mask,
                                    y_pred * mask,
                                    sample_weight)

class AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true_mask, y_pred, sample_weight=None):
        y_true, mask = tf.split(y_true_mask, 2, axis = -1)

        return super().update_state(y_true,
                                    y_pred,
                                    sample_weight)