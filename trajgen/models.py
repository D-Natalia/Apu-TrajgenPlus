import keras
import tensorflow as tf
from keras import Sequential
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def create_GRU_model(GRU_cells, 
                      seq_len, 
                      num_feat,
                      batch_size,
                      stateful,
                      return_seq,
                      num_outputs,
                      LR,
                      SEED,
                      ragged = False):
    """
        Create an GRU model with the specified parameters
        Returns the untrained model
    """
    
    keras.utils.set_random_seed(SEED)

    # In newer versions of Keras, for stateful LSTM, you need to specify the batch_input_shape as the first layer (input layer)
    model = Sequential()
    
    # Ragged tensor for variable length sequences
    if ragged is False:
        # if ragged is False, then the input layer should have a fixed batch size
        model.add(keras.layers.InputLayer(batch_input_shape=(batch_size, seq_len, num_feat)))

        # Iterate over the layers and add them to the model
        for i in range(len(GRU_cells) - 1):
            model.add(keras.layers.GRU(GRU_cells[i], return_sequences = return_seq, stateful = stateful))
            
        # Add the last layer
        model.add(keras.layers.GRU(GRU_cells[-1], return_sequences = return_seq, stateful = stateful))

    else:
        # if ragged is True, then the input layer should have a variable batch size
        model.add(keras.layers.InputLayer(shape=[None, num_feat], batch_size = batch_size, dtype=tf.float32, ragged = True))
        # Add the last layer
        model.add(keras.layers.GRU(GRU_cells, return_sequences = return_seq, stateful = False))
        
    # Add the final layer
    model.add(keras.layers.Dense(num_outputs))
    
    
    # https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
    # Create a learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = LR,
        decay_steps = 40,
        decay_rate = 0.96) 
    
    # Compile the model
    model.compile(optimizer = "adam", 
                  loss = "mse", 
                  metrics = ["mse", "mae", "mape", "kl_divergence"])
    
    # https://keras.io/api/optimizers/
    # Set the learning rate of the optimizer
    model.optimizer.lr=lr_schedule
    # mdl.optimizer.momentum = 0.99
    # mdl.optimizer.use_ema = True

    return model


def normalize_data(dataset, 
                   normalization_type = 'min-max',
                   normalization_ranges = None,
                   testing_data_norm = False):
    """
        Function to normalize the dataset using either min-max or standard normalization.
        Can be used for separate testing data normalization or for normalization of the whole dataset with other ranges.
        Returns the normalized dataset.
    """
    
    if normalization_type == 'min-max':   
        scaler = MinMaxScaler()
    elif normalization_type == 'standard':
        scaler = StandardScaler()
    
    # Normalize the dataset using the ranges given in normalization_ranges (min and max)        
    # Used for separate testing data normalization or for normalization of the whole dataset with other ranges
    
    if normalization_ranges is not None:
        scaler.min = normalization_ranges["min"]
        scaler.max = normalization_ranges["max"]
    else:
        scaler.fit(dataset)
        
    columns = dataset.columns
    
    norm_dataset = scaler.transform(dataset)
    norm_dataset = pd.DataFrame(norm_dataset, columns = columns)
    
    return scaler, norm_dataset
