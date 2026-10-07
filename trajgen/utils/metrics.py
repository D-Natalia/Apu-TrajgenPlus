import pandas as pd
import numpy as np

import utils.ldptrace

import utils.tsgan.dist_utils_tsgan as dst_utils_tsgan
import utils.ldptrace.grid as grid
import utils.ldptrace.map_func as map_func
import utils.ldptrace.trajectory as trajectory
import utils.ldptrace.experiment as experiment
import utils.ldptrace.utils as ldptrace_utils
from utils.ldptrace.grid import GridMap, Grid
from utils.ldptrace.experiment import SquareQuery

from typing import List, Tuple

from fastdtw import fastdtw

from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial.distance import euclidean
from scipy.stats import kendalltau
from scipy.spatial.distance import jensenshannon
from scipy.stats import energy_distance
from math import radians

def haversine_distance(lat1, lon1, lat2, lon2):
    """
        Compute great-circle (Haversine) distance between two lat/lon points in km.
        Returns the distance in km.
    """

    # Do this check if one argument is NaN
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0 # 
        
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    return haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371  #Earth radius in km

def haversine_distance_in_meters(lat1, lon1, lat2, lon2):
    """
        Compute great-circle (Haversine) distance between two lat/lon points in meters.
        Returns the distance in km.
    """
    
    return haversine_distance(lat1, lon1, lat2, lon2) * 1000  #Earth radius in meters


def compute_point_to_point_haversine_distances(traj1, traj2):
    """
    Compute the haversine point-to-point distance in meters between two trajectories.
    
    Parameters:
        traj1 (array-like): First trajectory as a list or array of [latitude, longitude] pairs.
        traj2 (array-like): Second trajectory as a list or array of [latitude, longitude] pairs.
    
    Returns:
        list: A list of distances in meters between corresponding points in the two trajectories.
    """
    if len(traj1) != len(traj2):
        raise ValueError("Trajectories must have the same number of points.")
    
    distances = []
    for (lat1, lon1), (lat2, lon2) in zip(traj1, traj2):
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        # Compute haversine distance in kilometers and convert to meters
        distance = haversine_distances([[lat1, lon1], [lat2, lon2]])[0, 1] * 6371000  # Earth radius in meters
        distances.append(distance)
    
    return distances

# ML metrics on normalized data
def compute_trajectory_metrics(X_test, Y_pred):
    
    """
        Compute prediction metrics for the model
        Returns a dictionary with the following metrics:
        - IE: Individual Error
        - ISE: Individual Squared Error
        - MSE: Mean Squared Error
        - ED: Energy Distance, averaged per features
    """
    
    errors = [0.0] * len(X_test)
    squared_errors = [0.0] * len(X_test)
    mses = [0.0] * len(X_test)
    eds = [0.0] * len(X_test)
    
    for i in range(len(X_test)):
        
        err = X_test[i] - Y_pred[i]
        squared_errors[i] = err**2
        errors[i] = err
        mses[i] = np.mean(err**2)
        
        eds1= float(energy_distance(X_test[i][:,0], Y_pred[i][:,0]))
        eds2 = float(energy_distance(X_test[i][:,1], Y_pred[i][:,1]))
        eds[i] = np.mean([eds1, eds2])
        
    results = {"IE" : errors, "ISE" : squared_errors, "MSE" : mses, "ED" : eds}
    
    return results

def exponential_moving_average_haversine(traj1, traj2, alpha=0.3):
    """
    Compute the Exponential Moving Average (EMA) of the point-to-point Haversine distances
    between two trajectories.

    Parameters:
        alpha (float): The smoothing factor, 0 < alpha <= 1. Higher alpha discounts older observations faster.

    Returns:
        list: The EMA values for the Haversine distances.
    """
    # Compute point-to-point Haversine distances
    distances = compute_point_to_point_haversine_distances(traj1, traj2)
    
    # Initialize EMA list
    ema = []
    
    # Compute EMA iteratively
    for i, dist in enumerate(distances):
        if i == 0:
            # First EMA value is the first distance
            ema.append(dist)
        else:
            # Apply EMA formula
            ema_value = alpha * dist + (1 - alpha) * ema[-1]
            ema.append(ema_value)
    
    return ema

def array_to_list(Y_test_dn: List[np.ndarray]) -> List[List[Tuple[float, float]]]:
    """
    Convert a list of trajectories (Y_test_dn) into a list of (lat, lon) tuples.

    Parameters:
        Y_test_dn (List[np.ndarray]): A list of trajectories, where each trajectory is a 2D array of [latitude, longitude].

    Returns:
        List[List[Tuple[float, float]]]: A list of trajectories, where each trajectory is a list of (lat, lon) tuples.
    """
    trajectories = []
    for traj in Y_test_dn:
        trajectory = [tuple(point) for point in traj]  # Convert each point to a (lat, lon) tuple
        trajectories.append(trajectory)
    return trajectories

def compute_min_max_coordinates(df):
    """
    Compute the minimum and maximum coordinates of a dataset of trajectories.

    Parameters:
        df (array-like): A list or array of trajectories, where each trajectory is a 2D array of [latitude, longitude].

    Returns:
        tuple: A tuple containing the minimum and maximum x (latitude) and y (longitude) coordinates, in the order (min_x, min_y, max_x, max_y).
    """
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for traj in df:
        # Extract the x (latitude) and y (longitude) values
        x_values = traj[:, 0]  # Latitude
        y_values = traj[:, 1]  # Longitude

        # Update the min and max values
        min_x = min(min_x, np.min(x_values))
        min_y = min(min_y, np.min(y_values))
        max_x = max(max_x, np.max(x_values))
        max_y = max(max_y, np.max(y_values))
    
    return min_x, min_y, max_x, max_y

def convert_raw_to_grid(raw_trajectories: List[List[Tuple[float, float]]], grid_map: GridMap,
                        interp=True):
    # Convert raw trajectories to grid trajectories
    """
    Convert raw trajectories to grid trajectories.

    Parameters:
        raw_trajectories (List[List[Tuple[float, float]]]): A list of raw trajectories, where each trajectory is a list of (lat, lon) tuples.
        grid_map (GridMap): The grid map to use for conversion.
        interp (bool, optional): Whether to interpolate the trajectory points to obtain the grid points. Defaults to True.

    Returns:
        List[List[Grid]]: A list of grid trajectories, where each trajectory is a list of Grid objects.
    """
    grid_db = [trajectory_point2grid(t, grid_map, interp)
               for t in raw_trajectories]
    return grid_db


def convert_grid_to_raw(grid_db: List[List[Grid]]):
    raw_trajectories = [trajectory.trajectory_grid2points(g_t) for g_t in grid_db]

    return raw_trajectories

def get_real_density(grid_db: List[List[Grid]], grid_map: GridMap):
    """
    Compute the real density from a list of grid trajectories and a grid map.

    Parameters:
        grid_db (List[List[Grid]]): A list of grid trajectories, where each trajectory is a list of Grid objects.
        grid_map (GridMap): The grid map to use for conversion.

    Returns:
        np.ndarray: A numpy array of shape (grid_size,) containing the real density of each grid point.
    """
    real_dens = np.zeros(grid_map.size)

    for t in grid_db:
        for g in t:
            index = map_func.grid_index_map_func(g, grid_map)
            real_dens[index] += 1

    return real_dens

def kl_divergence(prob1, prob2):
    """
    Compute the KL-divergence between two probability distributions.

    Parameters:
        prob1 (array-like): The first probability distribution.
        prob2 (array-like): The second probability distribution.

    Returns:
        float: The KL-divergence between the two distributions.
    """
    
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    kl = np.log((prob1 + 1e-8) / (prob2 + 1e-8)) * prob1

    return np.sum(kl)

def compute_js_distance_prob(prob1, prob2):
    """
    Compute the Jensen-Shannon distance between two probability distributions.

    Parameters:
        prob1 (array-like): The first probability distribution.
        prob2 (array-like): The second probability distribution.

    Returns:
        float: The Jensen-Shannon distance between the two distributions.
    """
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)

    avg_prob = (prob1 + prob2) / 2

    return 0.5 * kl_divergence(prob1, avg_prob) + 0.5 * kl_divergence(prob2, avg_prob)


def trajectory_point2grid(t: List[Tuple[float, float]], g: GridMap, interp=True):
    """
    Convert trajectory from raw points to grids
    :param t: raw trajectory
    :param g: grid map
    :param interp: whether to interpolate
    :return: grid trajectory
    """
    grid_map = g.map
    grid_t = list()

    for p in range(len(t)):
        point = t[p]
        found = False
        # Find which grid the point belongs to
        for i in range(len(grid_map)):
            for j in range(len(grid_map[i])):
                if grid_map[i][j].in_cell(point):
                    grid_t.append(grid_map[i][j])
                    found = True
                    break
            if found:
                break

    # Remove duplicates - does not work properly
    grid_t_new = [grid_t[0]]
    for i in range(1, len(grid_t)):
        if not grid_t[i].index == grid_t_new[-1].index:
            grid_t_new.append(grid_t[i])
    
    # # Remove duplicates 
    # seen_indices = set()
    # grid_t_new = []
    # for gt in grid_t:
    #     if gt.index not in seen_indices:
    #         grid_t_new.append(gt)
    #         seen_indices.add(gt.index)

    # Interpolation
    if interp:
        grid_t_final = list()
        for i in range(len(grid_t_new)-1):
            current_grid = grid_t_new[i]
            next_grid = grid_t_new[i+1]
            # Adjacent, no need to interpolate
            if grid.is_adjacent_grids(current_grid, next_grid):
                grid_t_final.append(current_grid)
            else:
                # Result of find_shortest_path() doesn't include the end point
                grid_t_final.extend(find_shortest_path(g, current_grid, next_grid))

        grid_t_final.append(grid_t_new[-1])
        return grid_t_final

    return grid_t_new

def find_shortest_path(g: GridMap, start: Grid, end: Grid):
    """
    Find the shortest path between two grids in a grid map using a simple iterative algorithm.

    Parameters:
        g (GridMap): The grid map to search.
        start (Grid): The starting grid.
        end (Grid): The ending grid.

    Returns:
        List[Grid]: The shortest path between the start and end grids. The end grid is not included in the path.

    Notes:
        This algorithm does not use any advanced search techniques and is not guaranteed to find the shortest path in all cases.
        The algorithm is simple and has a time complexity of O(n), where n is the number of grids in the map.
    """
    start_i, start_j = start.index
    end_i, end_j = end.index

    shortest_path = list()
    current_i, current_j = start_i, start_j

    while True:
        # NOTICE: shortest path doesn't include the end grid

        shortest_path.append(g.map[current_i][current_j])
        if end_i > current_i:
            current_i += 1
        elif end_i < current_i:
            current_i -= 1
        if end_j > current_j:
            current_j += 1
        elif end_j < current_j:
            current_j -= 1

        if end_i == current_i and end_j == current_j:
            break

    return shortest_path

def calculate_coverage_kendall_tau(orig_db: List[List[Grid]],
                                   syn_db: List[List[Grid]],
                                   grid_map: GridMap):
    """
    Compute the Kendall tau rank correlation coefficient between the coverage of original and synthetic trajectories.

    Parameters:
        orig_db (List[List[Grid]]): The original trajectory database.
        syn_db (List[List[Grid]]): The synthetic trajectory database.
        grid_map (GridMap): The grid map to use for conversion.

    Returns:
        float: The Kendall tau rank correlation coefficient between the coverage of original and synthetic trajectories.
    """
    actual_counts = np.zeros(grid_map.size)
    syn_counts = np.zeros(grid_map.size)

    # For each grid, find how many trajectories pass through it
    for i in range(len(grid_map.map)):
        for j in range(len(grid_map.map[0])):
            g = grid_map.map[i][j]
            index = map_func.grid_index_map_func(g, grid_map)
            for t in orig_db:
                actual_counts[index] += trajectory.pass_through(t, g)
            for t in syn_db:
                syn_counts[index] += trajectory.pass_through(t, g)

    concordant_pairs = 0
    reversed_pairs = 0
    for i in range(grid_map.size):
        for j in range(i + 1, grid_map.size):
            if actual_counts[i] > actual_counts[j]:
                if syn_counts[i] > syn_counts[j]:
                    concordant_pairs += 1
                else:
                    reversed_pairs += 1
            if actual_counts[i] < actual_counts[j]:
                if syn_counts[i] < syn_counts[j]:
                    concordant_pairs += 1
                else:
                    reversed_pairs += 1

    denominator = grid_map.size * (grid_map.size - 1) / 2
    return (concordant_pairs - reversed_pairs) / denominator

def compute_js_distance(traj1, traj2):
    """
    Compute the Jensen-Shannon distance between two trajectories.

    Parameters:
        traj1 (array-like): First trajectory as a list or array of points.
        traj2 (array-like): Second trajectory as a list or array of points.

    Returns:
        float: Jensen-Shannon distance between the two trajectories.
    """
    # Convert trajectories to probability distributions
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    
    # Normalize to ensure they sum to 1
    p = traj1 / np.sum(traj1)
    q = traj2 / np.sum(traj2)
    
    # Compute Jensen-Shannon distance
    js_distance = jensenshannon(p, q)
    return js_distance

def get_start_end_dist(grid_db: List[List[Grid]], grid_map: GridMap):
    """
    Compute the distribution of start and end points of all trajectories in a grid database.

    Parameters:
        grid_db (List[List[Grid]]): A list of grid trajectories, where each trajectory is a list of Grid objects.
        grid_map (GridMap): The grid map to use for conversion.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The distribution of start and end points. The first element of the tuple is a 2D numpy array of shape (grid_size, grid_size) where each element at index (i, j) represents the number of trajectories starting at grid i and ending at grid j. The second element is a 1D numpy array of shape (grid_size,) where each element at index i represents the number of trajectories starting at grid i. The third element is a 1D numpy array of shape (grid_size,) where each element at index i represents the number of trajectories ending at grid i.
    """
    dist = np.zeros(grid_map.size * grid_map.size)
    start_dist = np.zeros(grid_map.size)
    end_dist = np.zeros(grid_map.size)

    for g_t in grid_db:
        start = g_t[0]
        end = g_t[-1]
        index = map_func.pair_grid_index_map_func((start, end), grid_map)
        dist[index] += 1
        start_index = map_func.grid_index_map_func(start, grid_map)
        start_dist[start_index] += 1
        end_index = map_func.grid_index_map_func(end, grid_map)
        end_dist[end_index] += 1

    return dist, start_dist, end_dist


def compute_kl_divergence(traj1, traj2):
    """
    Compute the Kullback-Leibler (KL) divergence between two trajectories
    interpreted as probability distributions.

    Parameters:
        traj1 (array-like): First trajectory as a list or array of points.
        traj2 (array-like): Second trajectory as a list or array of points.

    Returns:
        float: KL divergence (D_KL(traj1 || traj2))
    """
    import numpy as np

    p = np.asarray(traj1, dtype=np.float64)
    q = np.asarray(traj2, dtype=np.float64)

    # Normalize to probability distributions
    p = p / (np.sum(p) + 1e-12)
    q = q / (np.sum(q) + 1e-12)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    kl_div = np.sum(p * np.log(p / q))
    return kl_div

def mutual_information_gps(traj_x, traj_y, n_lat_bins=20, n_lon_bins=20, base=np.e):
    """
    Mutual information between two GPS trajectories (each an (N,2) array of [lat,lon]),
    by discretizing each into a n_lat_bins × n_lon_bins grid.

    Parameters
    ----------
    traj_x, traj_y : array-like, shape (N, 2)
        Sequences of [latitude, longitude].
    n_lat_bins, n_lon_bins : int
        Number of bins along latitude and longitude.
    base : float
        Log base (e or 2).

    Returns
    -------
    I : float
        Estimated mutual information.
    """
    x = np.asarray(traj_x, dtype=float)
    y = np.asarray(traj_y, dtype=float)
    assert x.shape == y.shape and x.shape[1] == 2

    # build bin edges from the union of both trajectories
    lat_min = min(x[:,0].min(), y[:,0].min())
    lat_max = max(x[:,0].max(), y[:,0].max())
    lon_min = min(x[:,1].min(), y[:,1].min())
    lon_max = max(x[:,1].max(), y[:,1].max())

    lat_edges = np.linspace(lat_min, lat_max, n_lat_bins+1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon_bins+1)

    def to_cell_ids(traj):
        # digitize lat, lon separately then flatten to a single cell ID
        lat_idx = np.digitize(traj[:,0], bins=lat_edges) - 1
        lon_idx = np.digitize(traj[:,1], bins=lon_edges) - 1
        # clip any edge cases
        lat_idx = np.clip(lat_idx, 0, n_lat_bins-1)
        lon_idx = np.clip(lon_idx, 0, n_lon_bins-1)
        return lat_idx * n_lon_bins + lon_idx

    u = to_cell_ids(x)
    v = to_cell_ids(y)

    # return mutual_info_score(u, v)
    return mutual_information_discrete(u, v, base=base)

def mutual_information_discrete(u, v, base=np.e):
    """
    Compute I(u; v) for two discrete integer-valued sequences of the same length.
    """
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    assert u.shape == v.shape

    # joint histogram
    u_vals = np.unique(u)
    v_vals = np.unique(v)
    nu, nv = u_vals.size, v_vals.size

    # map values to 0…nu-1, 0…nv-1
    u_inv = np.searchsorted(u_vals, u)
    v_inv = np.searchsorted(v_vals, v)

    joint = np.zeros((nu, nv), dtype=float)
    for ui, vi in zip(u_inv, v_inv):
        joint[ui, vi] += 1
    joint /= joint.sum()

    # marginals (1D arrays)
    pu = joint.sum(axis=1)  # (nu,)
    pv = joint.sum(axis=0)  # (nv,)

    # Only consider nonzero joint probabilities
    nz = joint > 0
    nz_u, nz_v = np.where(nz)
    I = np.sum(
        joint[nz_u, nz_v] * np.log(joint[nz_u, nz_v] / (pu[nz_u] * pv[nz_v]))
    )
    if base != np.e:
        I /= np.log(base)
    return I


def mutual_information_gps(traj_x, traj_y, n_lat_bins=20, n_lon_bins=20, base=np.e):
    """
    Mutual information between two GPS trajectories (each an (N,2) array of [lat,lon]),
    by discretizing each into a n_lat_bins × n_lon_bins grid.

    Parameters
    ----------
    traj_x, traj_y : array-like, shape (N, 2)
        Sequences of [latitude, longitude].
    n_lat_bins, n_lon_bins : int
        Number of bins along latitude and longitude.
    base : float
        Log base (e or 2).

    Returns
    -------
    I : float
        Estimated mutual information.
    """
    x = np.asarray(traj_x, dtype=float)
    y = np.asarray(traj_y, dtype=float)
    assert x.shape == y.shape and x.shape[1] == 2

    # build bin edges from the union of both trajectories
    lat_min = min(x[:,0].min(), y[:,0].min())
    lat_max = max(x[:,0].max(), y[:,0].max())
    lon_min = min(x[:,1].min(), y[:,1].min())
    lon_max = max(x[:,1].max(), y[:,1].max())

    lat_edges = np.linspace(lat_min, lat_max, n_lat_bins+1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon_bins+1)

    def to_cell_ids(traj):
        # digitize lat, lon separately then flatten to a single cell ID
        lat_idx = np.digitize(traj[:,0], bins=lat_edges) - 1
        lon_idx = np.digitize(traj[:,1], bins=lon_edges) - 1
        # clip any edge cases
        lat_idx = np.clip(lat_idx, 0, n_lat_bins-1)
        lon_idx = np.clip(lon_idx, 0, n_lon_bins-1)
        return lat_idx * n_lon_bins + lon_idx

    u = to_cell_ids(x)
    v = to_cell_ids(y)

    # return mutual_info_score(u, v)
    return mutual_information_discrete(u, v, base=base)

def haversine_dist_in_km(a, b):
    return haversine_distance(a[0], a[1], b[0], b[1])

def compute_kendall_tau(traj1, traj2):
    """
    Compute the Kendall Tau correlation coefficient between two trajectories.

    Parameters:
        traj1 (array-like): First trajectory as a list or array of [x, y] or [latitude, longitude] points.
        traj2 (array-like): Second trajectory as a list or array of [x, y] or [latitude, longitude] points.

    Returns:
        float: Kendall Tau correlation coefficient.
        float: p-value for the test of non-correlation.
    """
    if len(traj1) != len(traj2):
        raise ValueError("Trajectories must have the same number of points.")
    
    # Flatten trajectories into a single dimension (e.g., distance from origin)
    traj1_ranks = np.argsort([np.linalg.norm(point) for point in traj1])
    traj2_ranks = np.argsort([np.linalg.norm(point) for point in traj2])
    
    # Compute Kendall Tau
    tau, p_value = kendalltau(traj1_ranks, traj2_ranks)
    return tau, p_value