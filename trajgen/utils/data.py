
import sys
sys.path.append('..')
from config import *

import copy

import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.metrics import *


def get_model_path(dataset):
    """
    Generate the file path for a model pickle file based on the dataset name.

    Args:
        dataset: The name of the dataset (string).

    Returns:
        The file path (string) to the model pickle file for the given dataset.

    This function constructs the model file path using the MODEL_FOLDER constant
    and the provided dataset name (converted to lowercase).
    """
    return MODEL_FOLDER + "mdl-" + dataset.lower() + ".pkl"

def save_pickle(data, data_path):
    """
    Save data to a pickle file.

    Args:
        data: The Python object to serialize and save.
        data_path: The file path where the pickle file will be written.

    This function opens the specified file in binary write mode and uses
    pickle.dump to serialize and save the provided data object.
    """
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
        
def load_pickle(data_path):
    """
    Load data from a pickle file.

    Args:
        data_path: The file path from which to load the pickle file.

    Returns:
        The Python object loaded from the pickle file.

    This function opens the specified file in binary read mode and uses
    pickle.load to deserialize and return the stored object.
    """
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    return df

def load_data_from_pickle(data_path, num_of_traj = 0):
    """
    Load data from a pickle file and optionally limit the number of trajectories.

    Args:
        data_path: The file path from which to load the pickle file.
        num_of_traj: The number of trajectories to load (default is 0, which loads all).

    Returns:
        The loaded data, limited to the specified number of trajectories if num_of_traj > 0.

    This function loads the data using load_pickle and slices it if a limit is provided.
    """
    df = load_pickle(data_path)
    if num_of_traj == 0:
        num_of_traj = len(df)
    df = df[:num_of_traj]
    return df

def create_X_Y_from_data(data, num_of_traj, k=1):
    """
        Create X and Y from the data.
        X is composed of the trajectory data starting from the first point to the second last point
        Y is composed of the trajectory data starting from the second point to the last point
        Keeps only the selected COLUMNS (defined in the parameters block)
    """
    X, Y = [0.0]  * num_of_traj, [0.0] * num_of_traj


    for i in range(num_of_traj):
        # X is composed of the trajectory data starting from the first point to the second last point
        X[i] =  data[i][COLUMNS_INPUT].iloc[0:-k] 
        X[i] = X[i].fillna(0)
        X[i].columns = COLUMNS_INPUT

        # Y is composed of the trajectory data starting from the second point to the last point
        Y[i] =  data[i][COLUMNS_OUTPUT].iloc[k:] 
        Y[i] = Y[i].fillna(0)
        Y[i].columns = COLUMNS_OUTPUT
        
        X[i] = X[i].to_numpy()
        Y[i] = Y[i].to_numpy()
        
    return X, Y

def get_min_max_from_data(data):
    """
    Get the minimum and maximum values from the data.

    The function goes through each trajectory in the data and finds the minimum and maximum values.
    It then takes the minimum and maximum of all the minimum and maximum values respectively,
    and returns them as a tuple.

    Args:
        data: A list of pandas DataFrames, each containing a trajectory.

    Returns:
        A tuple containing two numpy arrays, the first is the minimum values and the second is the maximum values.
    """
    # Get trajectories min and max values
    num_of_traj = len(data)
    mins, maxs = [0.0] * num_of_traj, [0.0] * num_of_traj

    # Loop through each trajectory and get the min and max values
    for i in range(num_of_traj):
        mins[i]  = np.array(data[i].min()) 
        maxs[i] = np.array(data[i].max()) 

    # Get the min and max of all the min and max values
    mins =  np.min( np.array(mins), axis = 0)[0 : 2]
    maxs =  np.max( np.array(maxs), axis = 0)[0 : 2]
    
    return mins, maxs


def get_data_in_square(data, square):
    """
    Get data inside the square defined by the square dictionary.

    The function loops through each trajectory in the data and checks if it is inside the square defined by the square dictionary.
    If the trajectory is inside the square, it is added to the filtered_data list which is then returned.

    Parameters:
        data: A list of pandas DataFrames, each containing a trajectory.
        square: A dictionary containing the coordinates of the square's corners.

    Returns:
        A list of pandas DataFrames, each containing a trajectory which is inside the square.
    """
    # Ensure correct bounds regardless of coordinate sign or order
    lat_min = min(square["lat_1"], square["lat_2"])
    lat_max = max(square["lat_1"], square["lat_2"])
    lon_min = min(square["lon_1"], square["lon_2"])
    lon_max = max(square["lon_1"], square["lon_2"])

    filtered_data = []
    
    # Loop through each trajectory in the data
    for traj in data:
        # Check if the trajectory is inside the square
        in_lat_bounds = traj["lat"].between(lat_min, lat_max)
        in_lon_bounds = traj["lon"].between(lon_min, lon_max)

        # If the trajectory is inside the square, add it to the filtered_data list
        if (in_lat_bounds & in_lon_bounds).all():
            filtered_data.append(traj)
            
            
    return filtered_data


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


def normalize_trajectory_data(dataset, 
                   normalization_type = 'min-max',
                   normalization_ranges = None,
                   testing_data_norm = False,
                   scaler = None):
    """
        Function to normalize the dataset using either min-max or standard normalization.
        Can be used for separate testing data normalization or for normalization of the whole dataset with other ranges.
        Returns the normalized dataset.
    """
    dataset_cpy = copy.deepcopy(dataset)
    
    if testing_data_norm is False:
        if scaler is None:
            if normalization_type == 'min-max':   
                scaler = MinMaxScaler()
            elif normalization_type == 'standard':
                scaler = StandardScaler()
        
        # Normalize the dataset using the ranges given in normalization_ranges (min and max)        
        # Used for separate testing data normalization or for normalization of the whole dataset with other ranges
        
        if normalization_ranges is not None:
            X_min = normalization_ranges["min"]
            X_max = normalization_ranges["max"]
            dataset_cpy = [(arr - X_min) / (X_max - X_min) for arr in dataset_cpy]
            
        else:
            dataset_flat = pd.concat(dataset_cpy, ignore_index=True)
            # dataset_flat = np.vstack(dataset_cpy)
            scaler.fit(dataset_flat)
            
            columns = dataset_cpy[0].columns
            
            for i in range(len(dataset_cpy)):
                norm_dataset = scaler.transform(dataset_cpy[i])
                norm_dataset = pd.DataFrame(norm_dataset, columns = columns)
                dataset_cpy[i] = norm_dataset
        
    else:
        
        if normalization_ranges is not None:
            X_min = normalization_ranges["min"]
            X_max = normalization_ranges["max"]
            dataset_cpy = [(arr - X_min) / (X_max - X_min) for arr in dataset_cpy]
            
        else:
            columns = dataset_cpy[0].columns
        
            for i in range(len(dataset_cpy)):
                norm_dataset = scaler.transform(dataset_cpy[i])
                norm_dataset = pd.DataFrame(norm_dataset, columns = columns)
                dataset_cpy[i] = norm_dataset
            
    return scaler, dataset_cpy


def add_speed_column(Y_pred_k, time_diff_seconds):
    """
    Adds a 'speed' column to each predicted trajectory in Y_pred_k, assuming constant time difference.
    Y_pred_k: list of np.ndarray, each of shape (n_points, 2) with columns [lat, lon]
    time_diff_seconds: scalar, time difference in seconds between each point (constant for all)
    Returns: list of np.ndarray, each of shape (n_points, 3) with columns [lat, lon, speed]
    """

    Y_pred_k_with_speed = []
    for traj in Y_pred_k:
        lats = traj[:, 0]
        lons = traj[:, 1]
        n = len(lats)
        lat_next = np.roll(lats, 1)
        lon_next = np.roll(lons, 1)
        lat_next[0] = 0.0
        lon_next[0] = 0.0

        distances = np.full(n, 0.0, dtype=np.float32)
        speeds = np.full(n, 0.0, dtype=np.float32)

        for i in range(1, n):
            distances[i] = haversine_distance(lats[i], lons[i], lat_next[i], lon_next[i])
            speeds[i] = distances[i] / (time_diff_seconds / 3600) if time_diff_seconds > 0 else 0

        traj_with_speed = np.column_stack((lats, lons, speeds))
        Y_pred_k_with_speed.append(traj_with_speed)
    return Y_pred_k_with_speed


def test_data_preparation(TRAINING_TESTING_SAME_FILE,
                          num_of_traj,
                          training_size,
                          SEQ_LEN,
                          NUM_FEATS,
                          TESTING_FILE,
                          data,
                          X=None, Y=None,
                          ):
    """
    Test data preparation
    Returns a list of numpy arrays, where each array is a single trajectory.
    """
    if TRAINING_TESTING_SAME_FILE:
        X_test = [0.0] * (num_of_traj - training_size)
        Y_test = [0.0] * (num_of_traj - training_size)

        test_traj_seq_lengths = [0.0] * (num_of_traj - training_size + 1)

        idx = 0
        for i in range(training_size, num_of_traj):
            # Convert the dataframe to numpy array
            X[i] = np.array(X[i])
            Y[i] = np.array(Y[i])

            # Get the sequence length for each trajectory
            test_traj_seq_lengths[idx] = X[i].shape[0]

            # Calculate the sequence multiplier and padding size
            seq_multiplier = X[i].shape[0] // SEQ_LEN
            padding_size = (seq_multiplier + 1) * SEQ_LEN - X[i].shape[0]

            # Create a padding of zeros
            padding = np.zeros([padding_size, NUM_FEATS])

            # Stack the padding and the trajectory
            X_test[idx] = np.vstack((X[i], padding))
            Y_test[idx] = Y[i]

            idx += 1
    else:
        # In case the testing data is obtained from a different file:
        data_test = load_data_from_pickle(TESTING_FILE)

        X_test = [0.0] * len(data_test)
        Y_test = [0.0] * len(data_test)

        # Get trajectories min and max values
        num_of_traj_test = len(data_test)

        test_traj_seq_lengths = [0.0] * num_of_traj_test

        data_test = [data_test[i][COLUMNS] for i in range(num_of_traj_test)]

        scaler, data_test = normalize_trajectory_data(dataset=data_test, normalization_type='min-max', testing_data_norm=True, scaler=scaler)

        X_t, Y_t = [0.0] * num_of_traj_test, [0.0] * num_of_traj_test

        for i in range(num_of_traj_test):
            # X is composed of the trajectory data starting from the first point to the second last point
            X_t[i] = data[i][COLUMNS_INPUT].iloc[0:-1]
            X_t[i] = X_t[i].fillna(0)
            X_t[i].columns = COLUMNS_INPUT

            # Y is composed of the trajectory data starting from the second point to the last point
            Y_t[i] = data[i][COLUMNS_OUTPUT].iloc[1:]
            Y_t[i] = Y_t[i].fillna(0)
            Y_t[i].columns = COLUMNS_OUTPUT

        for i in range(num_of_traj_test):
            X_t[i] = np.array(X_t[i])
            Y_t[i] = np.array(Y_t[i])

            test_traj_seq_lengths[i] = X[i].shape[0]

            # For data reshaping later on
            seq_multiplier = X_t[i].shape[0] // SEQ_LEN
            padding_size = (seq_multiplier + 1) * SEQ_LEN - X_t[i].shape[0]

            padding = np.zeros([padding_size, NUM_FEATS])

            X_test[i] = np.vstack((X_t[i], padding))
            Y_test[i] = Y_t[i]

    return X_test, Y_test, test_traj_seq_lengths


def denormalize_data(dataset, scaler = None, normalization_ranges = None):
    """
        Function to denormalize the dataset using the scaler used to normalize the dataset.
        Manual denormalization can be used for separate testing data denormalization or for denormalization of the whole dataset.
    """
    dataset_cpy = copy.deepcopy(dataset)
    
    #######
    if scaler is None and normalization_ranges is not None:
        X_min = normalization_ranges["min"]
        X_max = normalization_ranges["max"]
        
        dataset_cpy = [arr * (X_max - X_min) + X_min for arr in dataset]
            
    if scaler is not None:
        for item in range(len(dataset)):
            dataset_cpy[item] = scaler.inverse_transform(dataset_cpy[item])
       
    return dataset_cpy


def min_max_normalize_third_column(data):
    # Concatenate all third columns to find global min and max
    all_third = np.concatenate([traj.iloc[:, 2].values for traj in data])
    min_val = np.min(all_third)
    max_val = np.max(all_third)
    # Avoid division by zero
    if max_val == min_val:
        max_val += 1e-8

    # Normalize the third column for each trajectory
    normalized_data = []
    for traj in data:
        traj_copy = traj.copy()
        traj_copy.iloc[:, 2] = (traj_copy.iloc[:, 2] - min_val) / (max_val - min_val)
        normalized_data.append(traj_copy)
    return normalized_data

def add_speed_column(Y_pred_k, time_diff_seconds):
    """
    Adds a 'speed' column to each predicted trajectory in Y_pred_k, assuming constant time difference.
    Y_pred_k: list of np.ndarray, each of shape (n_points, 2) with columns [lat, lon]
    time_diff_seconds: scalar, time difference in seconds between each point (constant for all)
    Returns: list of np.ndarray, each of shape (n_points, 3) with columns [lat, lon, speed]
    """

    Y_pred_k_with_speed = []
    for traj in Y_pred_k:
        lats = traj[:, 0]
        lons = traj[:, 1]
        n = len(lats)
        lat_next = np.roll(lats, 1)
        lon_next = np.roll(lons, 1)
        lat_next[0] = 0.0
        lon_next[0] = 0.0

        distances = np.full(n, 0.0, dtype=np.float32)
        speeds = np.full(n, 0.0, dtype=np.float32)

        for i in range(1, n):
            distances[i] = haversine_distance(lats[i], lons[i], lat_next[i], lon_next[i])
            speeds[i] = distances[i] / (time_diff_seconds / 3600) if time_diff_seconds > 0 else 0

        traj_with_speed = np.column_stack((lats, lons, speeds))
        Y_pred_k_with_speed.append(traj_with_speed)
    return Y_pred_k_with_speed