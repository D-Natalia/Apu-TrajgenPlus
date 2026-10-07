'''
Continuous Trajectory Generation Based on Two-Stage GAN:
https://github.com/WenMellors/TS-TrajGen/blob/master/utils/evaluate_funcs.py

'''

import hausdorff as hausdorff_tsgan
from fastdtw import fastdtw
import math
import numba
import numpy as np
from math import sqrt, pow, cos, sin, asin
from inspect import getmembers
import utils.tsgan.distances_tsgan as distances

def hausdorff_metric(truth, pred, distance='haversine'):
    """豪斯多夫距离
    ref: https://github.com/mavillan/py-hausdorff

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    return hausdorff_distance(truth, pred, distance=distance)

def _hausdorff(XA, XB, distance_function):
	nA = XA.shape[0]
	nB = XB.shape[0]
	cmax = 0.
	for i in range(nA):
		cmin = np.inf
		for j in range(nB):
			d = distance_function(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	for j in range(nB):
		cmin = np.inf
		for i in range(nA):
			d = distance_function(XA[i,:], XB[j,:])
			if d<cmin:
				cmin = d
			if cmin<cmax:
				break
		if cmin>cmax and np.inf>cmin:
			cmax = cmin
	return cmax

def hausdorff_distance(XA, XB, distance='euclidean'):
	assert isinstance(XA, np.ndarray) and isinstance(XB, np.ndarray), \
		'arrays must be of type numpy.ndarray'
	assert np.issubdtype(XA.dtype, np.number) and np.issubdtype(XA.dtype, np.number), \
		'the arrays data type must be numeric'
	assert XA.ndim == 2 and XB.ndim == 2, \
		'arrays must be 2-dimensional'
	assert XA.shape[1] == XB.shape[1], \
		'arrays must have equal number of columns'
	
	if isinstance(distance, str):
		assert distance in _find_available_functions(distances), \
			'distance is not an implemented function'
		if distance == 'haversine':
			assert XA.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
			assert XB.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
		distance_function = getattr(distances, distance)
	elif callable(distance):
		distance_function = distance
	else:
		raise ValueError("Invalid input value for 'distance' parameter.")
	return _hausdorff(XA, XB, distance_function)

def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = (pow(math.sin(dlat/2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon/2.0), 2.0))
    return R * 2 * math.asin(math.sqrt(a))

def dtw_metric(truth, pred, distance='haversine'):
    """动态时间规整算法
    ref: https://github.com/slaypni/fastdtw

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    if distance == 'haversine':
        distance, path = fastdtw(truth, pred, dist=haversine)
    elif distance == 'manhattan':
        distance, path = fastdtw(truth, pred, dist=cityblock)
    elif distance == 'euclidean':
        distance, path = fastdtw(truth, pred, dist=euclidean)
    elif distance == 'chebyshev':
        distance, path = fastdtw(truth, pred, dist=chebyshev)
    elif distance == 'cosine':
        distance, path = fastdtw(truth, pred, dist=cosine)
    else:
        distance, path = fastdtw(truth, pred, dist=euclidean)
    return distance

def edit_distance(trace1, trace2, eps = 200):
    """
    the edit distance between two trajectory
    Args:
        trace1:
        trace2:
    Returns:
        edit_distance
    """
    matrix = [[i + j for j in range(len(trace2) + 1)] for i in range(len(trace1) + 1)]
    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            x0=trace1[i-1,0]
            y0=trace1[i-1,1]
            x1=trace2[j-1,0]
            y1=trace2[j-1,1]
            dx=(x0-x1)
            dy=(y0-y1)
            if sqrt(dx*dx+dy*dy) < eps:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(trace1)][len(trace2)]


def _find_available_functions(module_name):
	all_members = getmembers(module_name)
	available_functions = [member[0] for member in all_members 
						   if isinstance(member[1], numba.core.registry.CPUDispatcher)]
	return available_functions

