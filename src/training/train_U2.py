import sys
sys.path.append('src')
from models import sub_model
from utils import model_utils
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
import pandas as pd

tf.config.list_physical_devices('GPU')

model_save_path = "data/external/models" 
file_path = "data/processed"

data_x = pd.read_csv(file_path + "/all_U2_U12_x.txt.gz", compression='gzip', sep = " ", header = None)
data_x_5_dist = pd.read_csv(file_path + "/all_U2_U12_x_5_distance.txt.gz", compression='gzip', sep = " ", header = None)
data_x_3_dist = pd.read_csv(file_path + "/all_U2_U12_x_3_distance.txt.gz", compression='gzip', sep = " ", header = None)
data_y = pd.read_csv(file_path + "/all_U2_U12_y.txt.gz", compression='gzip', sep = " ", header = None)
data_type = pd.read_csv(file_path + "/all_U2_U12_info.txt.gz", compression='gzip', sep = " ", header = 0)
data_y_target = data_y.to_numpy()
data_y_target = tf.cast(data_y_target, dtype = tf.float32)

## OH encoding
data_x_oh = tf.convert_to_tensor(data_x.iloc[:,0:70].to_numpy(), dtype = tf.string)

string2int_layer = tf.keras.layers.StringLookup(vocabulary = ["A", "T", "C", "G"], output_mode='int')
data_x_oh = string2int_layer(data_x_oh)
data_x_oh = tf.one_hot(data_x_oh - 1, depth = 4, on_value = 1, off_value = 0)
data_x_oh = tf.cast(data_x_oh, dtype = tf.float32)
##

## Kmers encoding
data_x_kmers = model_utils.kmers_join(data_x, kmer = 3)

vocab_tmp = []
vocab = []
for one in "ATCGN":
    for two in "ATCGN":
        for three in "ATCGN":
            vocab_tmp.append(one + two + three)

for i in vocab_tmp:
    if i[1] != "N":
        if i[0] != "N" and i[2] != "N":
            vocab.append(i)
        if i[0] == "N" and i[2] != "N":
            vocab.append(i)
        if i[0] != "N" and i[2] == "N":
            vocab.append(i)

string2int_embed_layer = tf.keras.layers.StringLookup(vocabulary = vocab, output_mode='int')
data_x_kmers_embedding = string2int_embed_layer(data_x_kmers) - 1
##

## word2vec encoding
wv = KeyedVectors.load(model_save_path + "/word2vec_gencode_v19.3mers", mmap='r')
data_x_word2vec_embedding = []
for row in data_x_kmers:
    tmp = []
    for key in row:
        tmp.append(wv[key])
    data_x_word2vec_embedding.append(tmp)

data_x_word2vec_embedding = np.array(data_x_word2vec_embedding)
##

print(data_x_oh.shape)
print(data_x_kmers_embedding.shape)
print(data_x_word2vec_embedding.shape)
print(data_y_target.shape)

u2_index = (data_type.iloc[:,2] == "U2").to_numpy()
train_index = (data_type.iloc[:,3] == "train").to_numpy()
valid_index = (data_type.iloc[:,3] == "valid").to_numpy()
test_index = (data_type.iloc[:,3] == "test").to_numpy()

## train and valid
X_add_distance_train = np.concatenate((
                          data_x_5_dist.to_numpy().reshape((-1, 70, 1))[np.logical_and(train_index, u2_index)],
                          data_x_3_dist.to_numpy().reshape((-1, 70, 1))[np.logical_and(train_index, u2_index)]),
                          axis = -1)

X_add_distance_valid = np.concatenate((
                          data_x_5_dist.to_numpy().reshape((-1, 70, 1))[np.logical_and(valid_index, u2_index)],
                          data_x_3_dist.to_numpy().reshape((-1, 70, 1))[np.logical_and(valid_index, u2_index)]),
                          axis = -1)

X_train_oh = data_x_oh[np.logical_and(train_index, u2_index)]
X_train_kmers = data_x_kmers_embedding[np.logical_and(train_index, u2_index)]
X_train_word2vec = data_x_word2vec_embedding[np.logical_and(train_index, u2_index)]

X_valid_oh = data_x_oh[np.logical_and(valid_index, u2_index)]
X_valid_kmers = data_x_kmers_embedding[np.logical_and(valid_index, u2_index)]
X_valid_word2vec = data_x_word2vec_embedding[np.logical_and(valid_index, u2_index)]

y_train = tf.reshape(data_y_target, (-1, 70, 1))[np.logical_and(train_index, u2_index)]
y_valid = tf.reshape(data_y_target, (-1, 70, 1))[np.logical_and(valid_index, u2_index)]
##

## model training
oh_model = sub_model.base_oh_Dilated_CNN_LSTM(input_shape = (70, 4), distance_input_shape = (70, 2))
oh_model.compile(loss = "binary_crossentropy", 
                 metrics = [tf.keras.metrics.AUC(curve = 'PR', name = "auPRC"), tf.keras.metrics.AUC(curve = 'ROC', name = "auROC")],
                 optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001))

sub_model.ModelTrainer(oh_model).train((X_train_oh, X_add_distance_train), (X_valid_oh, X_add_distance_valid),
                                        y_train, y_valid, 
                                        PATIENCE = 15, EPOCHS = 1000, BATCH_SIZE = 32, VERBOSE = 1)

kmers_model = sub_model.base_kmers_embedding_Dilated_CNN_LSTM(input_shape = (70,), distance_input_shape = (70, 2), embedding_in = len(vocab), embedding_out = 100)
kmers_model.compile(loss = "binary_crossentropy", 
                    metrics = [tf.keras.metrics.AUC(curve = 'PR', name = "auPRC"), tf.keras.metrics.AUC(curve = 'ROC', name = "auROC")],
                    optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001))

sub_model.ModelTrainer(kmers_model).train((X_train_kmers, X_add_distance_train), (X_valid_kmers, X_add_distance_valid),
                                          y_train, y_valid, 
                                          PATIENCE = 15, EPOCHS = 1000, BATCH_SIZE = 32, VERBOSE = 1)

word2vec_model = sub_model.base_word2vec_embedding_Dilated_CNN_LSTM(input_shape = (70, 100), distance_input_shape = (70, 2))
word2vec_model.compile(loss = "binary_crossentropy", 
                       metrics = [tf.keras.metrics.AUC(curve = 'PR', name = "auPRC"), tf.keras.metrics.AUC(curve = 'ROC', name = "auROC")],
                       optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001))

sub_model.ModelTrainer(word2vec_model).train((X_train_word2vec, X_add_distance_train), (X_valid_word2vec, X_add_distance_valid),
                                              y_train, y_valid, 
                                              PATIENCE = 15, EPOCHS = 1000, BATCH_SIZE = 32, VERBOSE = 1)    
##

oh_model.save(model_save_path + '/all_U2_oh.h5')
kmers_model.save(model_save_path + '/all_U2_kmers.h5')
word2vec_model.save(model_save_path + '/all_U2_word2vec.h5')