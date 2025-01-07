import tensorflow as tf

def base_oh_simple_Dilated_CNN_LSTM(input_shape, distance_input_shape, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)
    distance_input_layer = tf.keras.Input(shape = distance_input_shape)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5])

    embedding = tf.keras.layers.Concatenate()([dcb1, distance_input_layer])  

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(embedding)
    # lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
    #                                      dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = (input_layer, distance_input_layer), outputs = output)

    return model

def base_solo_oh_Dilated_CNN_LSTM(input_shape, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5])

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(resblock)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = input_layer, outputs = output)

    return model

def base_oh_Dilated_CNN_LSTM(input_shape, distance_input_shape, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)
    distance_input_layer = tf.keras.Input(shape = distance_input_shape)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5])

    embedding = tf.keras.layers.Concatenate()([resblock, distance_input_layer])  

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(embedding)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = (input_layer, distance_input_layer), outputs = output)

    return model

def base_kmers_embedding_simple_Dilated_CNN_LSTM(input_shape, distance_input_shape, embedding_in, embedding_out, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)
    distance_input_layer = tf.keras.Input(shape = distance_input_shape)

    embedding = tf.keras.layers.Embedding(input_dim = embedding_in,
                                          output_dim = embedding_out)(input_layer)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(embedding)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(embedding)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5]) 

    embedding = tf.keras.layers.Concatenate()([dcb1, distance_input_layer])

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(embedding)
    # lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
    #                                      dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = (input_layer, distance_input_layer), outputs = output)

    return model    

def base_solo_kmers_embedding_Dilated_CNN_LSTM(input_shape, embedding_in, embedding_out, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)

    embedding = tf.keras.layers.Embedding(input_dim = embedding_in,
                                          output_dim = embedding_out)(input_layer)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(embedding)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(embedding)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5]) 

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(resblock)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = input_layer, outputs = output)

    return model

def base_kmers_embedding_Dilated_CNN_LSTM(input_shape, distance_input_shape, embedding_in, embedding_out, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)
    distance_input_layer = tf.keras.Input(shape = distance_input_shape)

    embedding = tf.keras.layers.Embedding(input_dim = embedding_in,
                                          output_dim = embedding_out)(input_layer)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(embedding)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(embedding)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5]) 

    embedding = tf.keras.layers.Concatenate()([resblock, distance_input_layer])

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(embedding)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = (input_layer, distance_input_layer), outputs = output)

    return model

def base_word2vec_embedding_simple_Dilated_CNN_LSTM(input_shape, distance_input_shape, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)
    distance_input_layer = tf.keras.Input(shape = distance_input_shape)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5])

    embedding = tf.keras.layers.Concatenate()([dcb1, distance_input_layer]) 

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(embedding)
    # lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
    #                                      dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = (input_layer, distance_input_layer), outputs = output)

    return model

def base_solo_word2vec_embedding_Dilated_CNN_LSTM(input_shape, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5])

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(resblock)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = input_layer, outputs = output)

    return model

def base_word2vec_embedding_Dilated_CNN_LSTM(input_shape, distance_input_shape, n_filter = 32, n_kernel = 7, n_unit = 32):

    input_layer = tf.keras.Input(shape = input_shape)
    distance_input_layer = tf.keras.Input(shape = distance_input_shape)

    dcb1 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb1 = tf.keras.layers.BatchNormalization()(dcb1)
    dcb1 = tf.keras.layers.ReLU()(dcb1)
    dcb1 = tf.keras.layers.Dropout(0.2)(dcb1)
    
    dcb2 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=2)(dcb1)
    dcb2 = tf.keras.layers.BatchNormalization()(dcb2)
    dcb2 = tf.keras.layers.ReLU()(dcb2)
    dcb2 = tf.keras.layers.Dropout(0.2)(dcb2)

    dcb3 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=4)(dcb2)
    dcb3 = tf.keras.layers.BatchNormalization()(dcb3)
    dcb3 = tf.keras.layers.ReLU()(dcb3)
    dcb3 = tf.keras.layers.Dropout(0.2)(dcb3)

    dcb4 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=8)(dcb3)
    dcb4 = tf.keras.layers.BatchNormalization()(dcb4)
    dcb4 = tf.keras.layers.ReLU()(dcb4)
    dcb4 = tf.keras.layers.Dropout(0.2)(dcb4)

    dcb5 = tf.keras.layers.Conv1D(n_filter, n_kernel, padding = "same", dilation_rate=1)(input_layer)
    dcb5 = tf.keras.layers.BatchNormalization()(dcb5)
    dcb5 = tf.keras.layers.ReLU()(dcb5)
    dcb5 = tf.keras.layers.Dropout(0.2)(dcb5)

    resblock = tf.keras.layers.Add()([dcb4, dcb5])

    embedding = tf.keras.layers.Concatenate()([resblock, distance_input_layer]) 

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(embedding)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_unit, return_sequences = True,
                                         dropout = 0.15, recurrent_dropout = 0))(lstm)
    lstm = SelfAttention(n_unit)(lstm)
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(lstm)

    model = tf.keras.Model(inputs = (input_layer, distance_input_layer), outputs = output)

    return model

class SelfAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    # self.layernorm = tf.keras.layers.LayerNormalization()
    # self.add = tf.keras.layers.Add()
    self.units = units

  def call(self, x):

    attn_output, attn_scores = self.mha(
        query=x,
        key=x,
        value=x,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    self.last_attention_weights = attn_scores

    # x = self.add([x, attn_output])
    # x = self.layernorm(x)

    # return x
    return attn_output
  
  def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config
  
class ModelTrainer:
    def __init__(self, model):
        self.model = model
        
    def train(self, X_train, X_valid,
              y_train, y_valid, 
              PATIENCE = 15, EPOCHS = 1000, BATCH_SIZE = 32, VERBOSE = 2):
        
        print (self.model.summary())

        callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = PATIENCE, restore_best_weights = True)
        self.model.fit(x = X_train, y = y_train, validation_data = (X_valid, y_valid), 
                       epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE,
                       callbacks = [callback])