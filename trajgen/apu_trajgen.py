
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from fastdtw import fastdtw
from config import *
from utils.data import *
from utils.metrics import *

def apu_trajgen_fixed_k(mdl,
                        X_t,
                        test_traj_seq_lengths,
                        SEQ_LENGTH,
                        NUM_FEATS,
                        k_steps=1
                        ):
    """
        Works for both stateful and non-stateful models
        Works for any batch size
        Predictions are made point-by-point
        Every k_steps, the model uses the real location
        Returns the predictions of the model on the test data
    """
    mdl.layers[0].reset_states()
    
    Y_preds = [0.0] * len(X_t)
 
    for i in range(len(X_t)):
        X_t[i] = X_t[i].reshape(-1, SEQ_LENGTH, NUM_FEATS)
    
    ## For each trajectory
    for i in range(len(X_t)):
        
        print("Processing trajectory:", i)
        
        # Use k_counter to keep track of the number of steps we have predicted wihtout using the real location
        k_counter = 0
        
        # Initialize Y_preds[i] as an empty list to store predictions step by step
        Y_preds[i] = []
        
        # For each point in the trajectory, we need to predict the next point
        for j in range(test_traj_seq_lengths[i]):
            
            # Reshape X_t to keep only the i, j position
            X_t_ij = X_t[i][j].reshape(1, SEQ_LENGTH, NUM_FEATS)  # Reshape to (1, 1, NUM_FEATS)
            
            # print("X_t_ij shape:", X_t_ij.shape)
        
            # If we are at the first point (k_counter == 0), we need to use the real location
            # or if we have predicted k steps, we need to use the real location
            # Otherwise, we use the predicted location
            if k_counter == 0 or k_counter >= k_steps:
                y_pred = mdl.predict(X_t_ij, batch_size = 1, verbose = 0)
                # Set k_counter to 1, to force the model to use the predicted location
                # for the next prediction (if k_steps > 1)
                k_counter = 1
            else:
                # Use the predicted location (from the previous step) for the next prediction
                y_p = X_t_ij
                y_p[:,:,0:2] = Y_preds[i][j-1][:,:]

                y_pred = mdl.predict(y_p.reshape(1, SEQ_LENGTH, NUM_FEATS), batch_size = 1, verbose = 0)
                # Increment k_counter
                k_counter += 1
        
            # Add the predicted location to the trajectory ("release the predicted location")
            Y_preds[i].append(y_pred)
        
        # Convert Y_preds[i] from a list to a NumPy array
        Y_preds[i] = np.array(Y_preds[i]).reshape(-1, NUM_OUTPUTS)
        
        mdl.layers[0].reset_states()
        
    return Y_preds

# The APU-TrajGen function with adaptive k (test_model_per_trajectory with adaptive k)
# The APU-TrajGen function with adaptive k (test_model_per_trajectory with adaptive k)
def apu_trajgen_adaptive_k(mdl, 
                            X_t, 
                            test_traj_seq_lengths,
                            SEQ_LENGTH,
                            NUM_FEATS,
                            k_steps=1,
                            su_funct = None,
                            su_funct_args = None,
                            normalization_ranges = None,
                            k_min = 1,
                            k_max = 5,
                            save_results = False,
                            save_step = 10,
                            save_name = None
                            ):
    """
        Works for both stateful and non-stateful models
        Works for any batch size
        Predictions are made point-by-point
        Depending on the SU score the prediction is reset to the real point or not
        Returns the predictions of the model on the test data
    """
    mdl.layers[0].reset_states()
    
    # Initialize k_counter_arr to keep track of the values of k_counter for each trajectory
    k_values = [0.0] * len(X_t)
    idx_reset = [0.0] * len(X_t)
    # k_min = 1
    
    Y_preds = [0.0] * len(X_t)

    for i in range(len(X_t)):
        X_t[i] = X_t[i].reshape(-1, SEQ_LENGTH, NUM_FEATS)
    
    ## For each trajectory
    for i in range(len(X_t)):
        
        # Use k_counter to keep track of the number of steps we have predicted wihtout using the real location
        k_counter = 0
        k_steps = k_min # 
        k_values[i] = []
        idx_reset[i] = []
        
        # Initialize Y_preds[i] as an empty list to store predictions step by step
        Y_preds[i] = []
        
        real_points = []
        pred_points = []
        
        # For each point in the trajectory, we need to predict the next point
        for j in range(test_traj_seq_lengths[i]):
            
            # Add k_counter to the trajectory
            k_values[i].append(k_steps)
        
            # Reshape X_t to keep only the i, j position
            X_t_ij = X_t[i][j].reshape(1, SEQ_LENGTH, NUM_FEATS)  # Reshape to (1, SEQ_LENGTH1, NUM_FEATS)
            
            # If we are at the first point (k_counter == 0), we need to use the real location
            # or if we have predicted k steps, we need to use the real location
            # Otherwise, we use the predicted location
            if (j>0 and su_funct != None and su_funct_args != None and normalization_ranges != None):
                
                # Denormalize the real point and the predicted point to compute the SU score
                x_t_dn = denormalize_data(X_t_ij[:,:,0:2], normalization_ranges=normalization_ranges)
                y_pred_dn = denormalize_data(Y_preds[i][j-1], normalization_ranges=normalization_ranges) #use the previous predicted point
                
                real_points.append(x_t_dn)
                pred_points.append(y_pred_dn)
                
                su_funct_args["real_points"] = real_points
                su_funct_args["pred_points"] = pred_points
                # y_pred_dn = denormalize_data(y_pred, scaler) #use the previous predicted point
                
                # Compute the distance between the real point and the predicted point
                dist = haversine_distance_in_meters(x_t_dn[0][0][0], x_t_dn[0][0][1], y_pred_dn[0][0][0], y_pred_dn[0][0][1])
                
                # Compute the SU score
                # The SU score is computed using the denormalized data
                su_score = su_funct(x_t_dn, y_pred_dn, su_funct_args)
                
                # Apply the rules to update k_steps (see the paper for details)
                if (su_score < 0):
                    k_steps += 1
                    if (k_steps > k_max):
                        k_steps = k_max
                elif (su_score > 1):
                    k_steps -= 1
                    if (k_steps < k_min):
                        k_steps = k_min
                # if (su_score < 0 and k_steps < k_max):
                #     k_steps += 1
                # elif (su_score > 1):
                #     if (k_steps == 1):
                #         k_steps = k_min
                #     else:
                #         k_steps -= 1
                    
            # If we are at the first point (k_counter == 0), we need to use the real location
            # Or if the counter needs to be reset (k_counter >= k_steps), we need to use the real location
            # Otherwise, we use the predicted location
            if k_counter == 0 or k_counter >= k_steps:
                x_t = X_t_ij
                idx_reset[i].append(1)
                # Set k_counter to 1
                k_counter = 1
            else:
                # Use the predicted location (from the previous step) for the next prediction
                x_t = X_t_ij
                x_t[:,:,0:2] = Y_preds[i][j-1][:,:]
                idx_reset[i].append(0)
            
            # Predict the next point    
            y_pred = mdl.predict(x_t, batch_size = 1, verbose = 0)
                        
            k_counter += 1
        
            # Add the predicted location to the trajectory ("release the predicted location")
            Y_preds[i].append(y_pred)
        
        # Convert Y_preds[i] from a list to a NumPy array
        Y_preds[i] = np.array(Y_preds[i]).reshape(-1, NUM_OUTPUTS)
        
        if (save_results == True and save_name is not None and (i+1) % save_step == 0):
            # Save the predictions to a pickle file
            save_pickle(Y_preds[i*save_step : (i+1)*save_step], save_name + "_" + str(round((i+1) / save_step * save_step)) + ".pkl")
        
        mdl.layers[0].reset_states()
        
    return Y_preds, k_values, idx_reset

# Function that computes the SU score for two partial trajectories
# SU score = (distance - mean_min) / (mean_max - mean_min)
def compute_su_score1(real_point, pred_point, su_funct_args):
    """
        Compute the SU score for two partial trajectories
        In this case the score computes the distance between the two points
        and normalizes it using the selected mean values (see the paper for details)
    """
    
    # Extract latitude and longitude
    lat1, lon1 = real_point[0][0][0], real_point[0][0][1]
    lat2, lon2 = pred_point[0][0][0], pred_point[0][0][1]

    # Compute the haversine distance
    dist = haversine_distance_in_meters(lat1, lon1, lat2, lon2)
    
    # Compute the SU score
    score = (dist - su_funct_args['mean_min']) / (su_funct_args['mean_max'] - su_funct_args['mean_min'])
    
    return score

# Function that computes the SU score for two partial trajectories using arithmetic mean
def compute_su_score2(real_point, pred_point, su_funct_args):
    """
    Compute the SU score for two partial trajectories.
    Calculates the point-to-point Haversine distance along the sequences,
    computes the arithmetic mean of these distances, and normalizes it
    using the provided mean bounds.
    """
    # Extract historical point sequences
    real_points = su_funct_args["real_points"]
    pred_points = su_funct_args["pred_points"]
    
    # Calculate point-to-point distances
    distance_point_to_point=[]
    for i in range (len(real_points)):
        lat1,lon1=real_points[i][0][0][0], real_points[i][0][0][1]
        lat2,lon2=pred_points[i][0][0][0], pred_points[i][0][0][1]
        d=haversine_distance_in_meters(lat1,lon1,lat2,lon2)
        distance_point_to_point.append(d)
    
    # Compute arithmetic mean distance
    dist=np.mean(distance_point_to_point)

    # Compute the SU score
    score = (dist - su_funct_args['mean_min']) / (su_funct_args['mean_max'] - su_funct_args['mean_min'])
    
    return score

# Function that computes the SU score for two partial trajectories using EMA
def compute_su_score3(real_point, pred_point, su_funct_args):
    """
    Compute the SU score for two partial trajectories using an Exponential Moving Average (EMA).
    Calculates step-by-step Haversine distances and applies EMA smoothing to weight
    recent trajectory errors more heavily than older ones, then normalizes the result.
    """
    # Extract historical point sequences
    real_points = su_funct_args["real_points"]
    pred_points = su_funct_args["pred_points"]
    
    # Smoothing factor: higher value gives more weight to the most recent points
    alpha=0.25

    ema_dist=None
    distance_point_to_point=[]

    # Calculate distances and update EMA sequentially
    for i in range (len(real_points)):
        lat1,lon1=real_points[i][0][0][0], real_points[i][0][0][1]
        lat2,lon2=pred_points[i][0][0][0], pred_points[i][0][0][1]
        d=haversine_distance_in_meters(lat1,lon1,lat2,lon2)

        # Initialize with first point distance, then apply exponential smoothing
        if ema_dist is None:
            ema_dist = d
        else:
            ema_dist = alpha * d + (1 - alpha) * ema_dist

    # Compute the SU score
    score = (ema_dist - su_funct_args['mean_min']) / (su_funct_args['mean_max'] - su_funct_args['mean_min'])
    
    return score

# Function that computes the SU score for two partial trajectories using Dynamic Time Warping (DTW)
def compute_su_score5(real_point, pred_point, su_funct_args):
    """
    Compute the SU score for two partial trajectories using Dynamic Time Warping (DTW).
    Aligns trajectories using the FastDTW algorithm with Haversine distance as the local
    cost metric, computes the average distance along the optimal warp path, and normalizes it.
    """
    # Extract historical point sequences
    real_points = su_funct_args["real_points"]
    pred_points = su_funct_args["pred_points"]
    
    # Local pairwise metric for DTW alignment
    def dtw_metric(p1, p2):
        lat1, lon1 = p1[0][0][0], p1[0][0][1]
        lat2, lon2 = p2[0][0][0], p2[0][0][1]
        return haversine_distance_in_meters(lat1, lon1, lat2, lon2)

    # Compute DTW cumulative distance and optimal alignment path
    distance_dwt, path = fastdtw(real_points,pred_points, dist=dtw_metric)

    # Normalize cumulative alignment cost by path length to get average point error
    dist_mean=distance_dwt/len(path)

    # Compute the SU score
    score = (dist_mean - su_funct_args['mean_min']) / (su_funct_args['mean_max'] - su_funct_args['mean_min'])

    return score


