

# Paths

MODEL_FOLDER = "../models/"
DATA_FOLDER = "../data/" 

# Configuration file for trajectory generation

SEED = 123456
TESTING_SIZE = 0.2

# Total number of trajectories. If set to 0, all trajectories are used
TOTAL_TRAJS = 0

TRAINING_TESTING_SAME_FILE = True

DATASET = {"PORTO": "../datasets/Porto/porto_uci_31k_traj_time_diff.pkl",
           "SANFRANCISCO": "../datasets/SanFrancisco/train_trajectories_time_diff.pkl"}

TESTING_FILE = None

COLUMNS = ["lat", "lon", "speed_km"]

COLUMNS_INPUT = ["lat", "lon", "speed_km"]
COLUMNS_OUTPUT = ["lat", "lon"]

DATA_SQUARE_SF = { 
                "lat_1": 37.86499,
                "lon_1": -122.53304,
                "lat_2": 37.68481,
                "lon_2": -122.30576
                }

DATA_SQUARE_PORTO = {
                "lat_1": 41.23969,
                "lon_1": -8.73005,
                "lat_2": 41.05951,
                "lon_2": -8.49195
                }

DATA_SQUARE = {"SANFRANCISCO": DATA_SQUARE_SF,
               "PORTO": DATA_SQUARE_PORTO}

DATA_CENTER_PORTO = {
    "lat": 41.14961, # Latitude of Porto
    "lon": -8.61099 # Longitude of Porto
}

DATA_CENTER_SF = {
    "lat": 37.7749,  # Latitude of San Francisco
    "lon": -122.4194  # Longitude of San Francisco
}

MAX_K = 1

# Model parameters
LSTM_CELLS = [8] #[16, 8, 16] #32 #32
SEQ_LEN = 25 #25
BATCH_SIZE = 64 #32 #32
EPOCHS = 100
LR = 0.01

STATEFUL = False
RETURN_SEQ = True

NUM_FEATS = 3
NUM_OUTPUTS = 2

FEATS = [0, 1, 2] # lat, lon, speed_km
OUTPUTS = [0, 1] # lat, lon

# The plotting parameters
PLOT_FONT_SMALL_SIZE = 16
PLOT_FONT_MEDIUM_SIZE = 20
PLOT_FOLDER = "../plots/"
    
PLOT_FONT = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 5,
    }