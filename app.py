import sys
import os
import time
import urllib.request
import json
import math
import pickle
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import tensorflow.keras as keras

# Adauga calea pentru importuri locale
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajgen'))

from config import SEQ_LEN, NUM_FEATS, DATA_SQUARE_PORTO, DATA_CENTER_PORTO
from utils.metrics import haversine_distance_in_meters
# Nu mai folosim funcțiile din utils/data.py deoarece sunt optimizate pentru preprocesarea DataFrame-urilor
from trajgen.apu_trajgen import compute_su_score1, compute_su_score2, compute_su_score3

app = FastAPI(title="APU-TrajGen Obfuscator API")

keras_model = None
try:
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mdlgru-porto.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            keras_model = pickle.load(f)
        print(f"Modelul Keras (pickle) a fost încărcat cu succes din {pkl_path}!")
        keras_model.summary()
    else:
        print(f"Atenție: Fișierul modelului nu a fost găsit la calea: {pkl_path}. Backend-ul rulează în modul fallback.")
except Exception as e:
    print(f"Atenție: Nu am putut încărca modelul Keras ({e}). Backend-ul rulează în modul fallback.")

# Structura cererii
class LocationRequest(BaseModel):
    user_id: str
    lat: float
    lng: float
    timestamp: float = None    
    su_method: str = "1"        
    protection_interval: str = "80-155" 


USER_STATES = {}

class RoutePoint(BaseModel):
    lat: float
    lon: float
    timestamp: float = None

class RouteRequest(BaseModel):
    user_id: str
    route: List[RoutePoint]
    su_method: str = "1"
    protection_interval: str = "80-155"


K_MIN = 2
K_MAX = 4

PROTECTION_INTERVALS = {
    "80-155": {'mean_min': 80.0, 'mean_max': 155.0},
    "170-350": {'mean_min': 170.0, 'mean_max': 350.0},
    "250-450": {'mean_min': 250.0, 'mean_max': 450.0},
    "350-550": {'mean_min': 350.0, 'mean_max': 550.0},
}
SU_FUNCT_ARGS = {'mean_min': 80.0, 'mean_max': 155.0} 

NORMALIZATION_RANGES = {
    "min": np.array([
        min(DATA_SQUARE_PORTO["lat_1"], DATA_SQUARE_PORTO["lat_2"]),
        min(DATA_SQUARE_PORTO["lon_1"], DATA_SQUARE_PORTO["lon_2"]),
        0.0   # Viteza minimă (km/h)
    ]),
    "max": np.array([
        max(DATA_SQUARE_PORTO["lat_1"], DATA_SQUARE_PORTO["lat_2"]),
        max(DATA_SQUARE_PORTO["lon_1"], DATA_SQUARE_PORTO["lon_2"]),
        120.0
    ])
}

RETURN_FULL_ROUTE = True
MAX_ALLOWED_PRED_DISTANCE = 5000.0  # metri

def snap_to_road(lat, lng):
    """
    Folosește API-ul public gratuit OSRM pentru a 'lipi' coordonatele de cea mai apropiată stradă.
    """
    try:
        url = f"http://router.project-osrm.org/nearest/v1/driving/{lng},{lat}?number=1"
        orsm_req = urllib.request.Request(url, headers={'User-Agent': 'APU-TrajGen-MVP/1.0'})
        
        with urllib.request.urlopen(orsm_req, timeout=30) as response:
            data = json.loads(response.read().decode())
            if data.get("code") == "Ok" and len(data.get("waypoints", [])) > 0:
                snapped_lng, snapped_lat = data["waypoints"][0]["location"]
                return snapped_lat, snapped_lng
    except Exception as e:
        print(f"OSRM snap error: {e}")
    
    return lat, lng

def get_route_between_points(start_lat, start_lon, end_lat, end_lon):
    """
    Obține ruta rutieră completă între două puncte folosind OSRM Route API.
    Returnează o listă de dicționare [{"lat": lat, "lon": lon}, ...]
    """
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=simplified&geometries=geojson"
        req = urllib.request.Request(url, headers={'User-Agent': 'APU-TrajGen-MVP/1.0'})
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                coordinates = data["routes"][0]["geometry"]["coordinates"]
                return [{"lat": coord[1], "lon": coord[0]} for coord in coordinates]
    except Exception as e:
        print(f"OSRM route error: {e}")
    
    return [{"lat": end_lat, "lon": end_lon}]

def calculate_speed_kmh(lat1, lon1, lat2, lon2, time_diff_sec):
    if time_diff_sec <= 0:
        return 0.0
    dist_m = haversine_distance_in_meters(lat1, lon1, lat2, lon2)
    dist_km = dist_m / 1000.0
    time_hours = time_diff_sec / 3600.0
    return dist_km / time_hours

@app.post("/obfuscate")
async def obfuscate_location(req: LocationRequest, return_full_route: bool = True):
    if keras_model is None:
        pass

    start_time = time.time() 

    user_id = req.user_id
    current_time = req.timestamp or time.time()


    if user_id not in USER_STATES:

        offset_lat = req.lat - DATA_CENTER_PORTO["lat"]
        offset_lon = req.lng - DATA_CENTER_PORTO["lon"]
        
        USER_STATES[user_id] = {
            "sequence": [],
            "k_steps": K_MIN,        
            "k_counter": 0,          
            "last_pred": None,        
            "last_pred_porto": None, 
            "last_real": None,        
            "last_time": current_time,
            "offset_lat": offset_lat, 
            "offset_lon": offset_lon, 
            "real_points_history": [],
            "pred_points_history": [],
            "eval_history": [],
            "total_distance_error": 0.0, 
            "point_count": 0          
        }

    state = USER_STATES[user_id]
    offset_lat = state["offset_lat"]
    offset_lon = state["offset_lon"]

    porto_lat = req.lat - offset_lat
    porto_lon = req.lng - offset_lon

    speed_kmh = 0.0
    if state["last_real"] is not None:
        time_diff = current_time - state["last_time"]
        prev_lat, prev_lon = state["last_real"]
        speed_kmh = calculate_speed_kmh(prev_lat, prev_lon, req.lat, req.lng, time_diff)

    su_score = None
    if state["last_pred"] is not None:
        current_real = (req.lat, req.lng)
        current_pred = state["last_pred"]
        
        dist_error = haversine_distance_in_meters(current_real[0], current_real[1], current_pred[0], current_pred[1])
        state["total_distance_error"] += dist_error
        state["point_count"] += 1
        mde = state["total_distance_error"] / state["point_count"] if state["point_count"] > 0 else 0

        eval_history = state.get("eval_history", [])
        real_seq = [pair[0] for pair in eval_history] + [current_real]
        pred_seq = [pair[1] for pair in eval_history] + [current_pred]
        
        real_arr = np.array([[real_seq[-1]]]) 
        pred_arr = np.array([[pred_seq[-1]]]) 
        
        su_funct_args = PROTECTION_INTERVALS.get(req.protection_interval, SU_FUNCT_ARGS)
        current_args = su_funct_args.copy()
        current_args["real_points"] = [np.array([[[lat, lng]]]) for lat, lng in real_seq]
        current_args["pred_points"] = [np.array([[[lat, lng]]]) for lat, lng in pred_seq]

        if req.su_method == "2":
            su_score = compute_su_score2(None, None, current_args)
        elif req.su_method == "3":
            su_score = compute_su_score3(None, None, current_args)
        else: # Default to SU1
            su_score = compute_su_score1(real_arr, pred_arr, su_funct_args)

        if su_score is not None:
            k_before = state["k_steps"]
            if su_score < 0:
                state["k_steps"] = min(state["k_steps"] + 1, K_MAX)
            elif su_score > 1:
                state["k_steps"] = max(state["k_steps"] - 1, K_MIN)
            k_after = state["k_steps"]
            print(f"[DEBUG] su_method={req.su_method} su_score={su_score:.4f} k_steps={k_before}->{k_after} MDE={mde:.2f}m")
                
        if "eval_history" not in state:
            state["eval_history"] = []
        state["eval_history"].append((current_real, current_pred))
        
        if len(state["eval_history"]) > 3:
            state["eval_history"].pop(0)

    input_features = [porto_lat, porto_lon, speed_kmh]

    if state["k_counter"] == 0 or state["k_counter"] >= state["k_steps"]:
        
        state["k_counter"] = 1
    else:
        
        if state["last_pred_porto"] is not None:
            input_features[0] = state["last_pred_porto"][0]
            input_features[1] = state["last_pred_porto"][1]
        state["k_counter"] += 1

    X_min = NORMALIZATION_RANGES["min"]
    X_max = NORMALIZATION_RANGES["max"]
    
    range_diff = X_max - X_min
    range_diff[range_diff == 0] = 1.0 

    input_features_norm = (np.array(input_features) - X_min) / range_diff
    
    state["sequence"].append(input_features_norm)
    if len(state["sequence"]) > SEQ_LEN:
        state["sequence"].pop(0) 

    seq_array = np.array(state["sequence"])
    if len(seq_array) < SEQ_LEN:
        pad_size = SEQ_LEN - len(seq_array)
        seq_array = np.pad(seq_array, ((pad_size, 0), (0, 0)), mode='edge')
    
    model_input_norm = seq_array.reshape(1, SEQ_LEN, NUM_FEATS).astype(np.float32)

    if keras_model is not None:
        pred_norm = keras_model.predict(model_input_norm, batch_size=1, verbose=0)
        
        last_pred_norm = pred_norm[0][-1]
        pred_denorm = last_pred_norm * (X_max[:2] - X_min[:2]) + X_min[:2]
        
        pred_porto_lat, pred_porto_lon = float(pred_denorm[0]), float(pred_denorm[1])
    else:
        if len(state["sequence"]) > 1:
            last_point = state["sequence"][-1]
            prev_point = state["sequence"][-2]
            delta = last_point - prev_point
            pred_norm = last_point + delta
            pred_denorm = pred_norm[:2] * (X_max[:2] - X_min[:2]) + X_min[:2]
            pred_porto_lat, pred_porto_lon = float(pred_denorm[0]), float(pred_denorm[1])
        else:
            pred_porto_lat, pred_porto_lon = porto_lat + 0.001, porto_lon + 0.001
        
    state["last_pred_porto"] = (pred_porto_lat, pred_porto_lon)

    pred_real_lat = pred_porto_lat + offset_lat
    pred_real_lon = pred_porto_lon + offset_lon

    MIN_PRIVACY_DIST = 30.0  
    dist_to_real = haversine_distance_in_meters(req.lat, req.lng, pred_real_lat, pred_real_lon)
    
    if dist_to_real < MIN_PRIVACY_DIST:
        if dist_to_real < 1.0: 
            pred_real_lat += 0.001
            pred_real_lon += 0.001
        else:
            ratio = MIN_PRIVACY_DIST / dist_to_real
            pred_real_lat = req.lat + (pred_real_lat - req.lat) * ratio
            pred_real_lon = req.lng + (pred_real_lon - req.lng) * ratio

    if return_full_route and state["last_pred"] is not None:
            last_lat, last_lon = state["last_pred"]
            obfuscated_route = get_route_between_points(last_lat, last_lon, pred_real_lat, pred_real_lon)
    else:
            snap_lat, snap_lon = snap_to_road(pred_real_lat, pred_real_lon)
            obfuscated_route = [{"lat": snap_lat, "lon": snap_lon}]

    

    state["last_real"] = (req.lat, req.lng)
    state["last_time"] = current_time
    if len(obfuscated_route) > 0:
        state["last_pred"] = (obfuscated_route[-1]["lat"], obfuscated_route[-1]["lon"])
    else:
        state["last_pred"] = (pred_real_lat, pred_real_lon)
        obfuscated_route = [{"lat": pred_real_lat, "lon": pred_real_lon}]
    
    exec_time_ms = (time.time() - start_time) * 1000
    print(f"[DEBUG] Time: {exec_time_ms:.2f}ms | Interval: {req.protection_interval} | Dist: {dist_to_real:.2f}m | SU: {su_score if su_score is not None else 'N/A'}")

    return {
        "user_id": user_id,
        "obfuscated_route": obfuscated_route,
        "generated_point": {"lat": pred_real_lat, "lon": pred_real_lon},
        "k_steps": state["k_steps"],
        "su_score": su_score
    }

@app.post("/obfuscate_route")
async def obfuscate_full_route(body: RouteRequest):
    user_id = body.user_id
    
    if user_id in USER_STATES:
        del USER_STATES[user_id]
        
    synthetic_points_for_osrm = []
    raw_generated_points = []
    final_k_steps = K_MIN
    start_time = time.time()
    
    for point in body.route:
        point_req = LocationRequest(
            user_id=user_id,
            lat=point.lat,
            lng=point.lon,
            timestamp=point.timestamp or time.time(),
            su_method=body.su_method,
            protection_interval=body.protection_interval
        )
        res = await obfuscate_location(point_req, return_full_route=False)
        
        if res.get("generated_point"):
            raw_generated_points.append(res["generated_point"])
            synthetic_points_for_osrm.append(res["generated_point"])

        final_k_steps = res["k_steps"]

    full_obfuscated_route = []
    if len(synthetic_points_for_osrm) > 1:
        coords_str = ";".join([f"{p['lon']},{p['lat']}" for p in synthetic_points_for_osrm]) 
        url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'APU-TrajGen-MVP/1.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode())
                if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                    coordinates = data["routes"][0]["geometry"]["coordinates"]
                    full_obfuscated_route = [{"lat": coord[1], "lon": coord[0]} for coord in coordinates]
        except Exception as e:
            print(f"OSRM bulk route error: {e}")
            full_obfuscated_route = synthetic_points_for_osrm
    elif len(synthetic_points_for_osrm) == 1:
        full_obfuscated_route = synthetic_points_for_osrm

    exec_time_ms = (time.time() - start_time) * 1000
    print(f"[DEBUG] Full route regenerated in {exec_time_ms:.2f}ms for {len(body.route)} points.")

    return {
        "user_id": user_id,
        "obfuscated_route": full_obfuscated_route,
        "generated_route": raw_generated_points, 
        "k_steps": final_k_steps
    }
