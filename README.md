# APU-TrajGenPlus

Extension of **APU-TrajGen+** with additional utility evaluation methods, experimental analysis, and a mobile application for privacy-preserving synthetic GPS trajectories.

## 📌 About the Project

This repository contains my work developed as an extension of the original **APU-TrajGen+** project.

APU-TrajGen+ is a privacy-preserving synthetic trajectory generation framework based on a GRU neural network and adaptive trajectory generation. The original project was developed by its respective authors and is available here:

**Original project:**
https://github.com/aromanrsc/APU-TrajGenPlus

This repository focuses on the extensions and experiments developed as part of my bachelor's thesis.

---

## 👩‍💻 My Contributions

The main contributions developed in this project are:

### 1. Additional Utility Score Methods

Three additional utility evaluation methods were implemented in:

`trajgen/apu_trajgen.py`

#### SU2 — Arithmetic Mean Haversine Distance

Computes the arithmetic mean of the point-to-point Haversine distances between the real and synthetic trajectories.

The resulting value is normalized using the minimum and maximum mean-distance values.

#### SU3 — Exponential Moving Average

Uses an Exponential Moving Average (EMA) to evaluate trajectory similarity.

The implementation uses:

```text
alpha = 0.25
```

The method processes the Haversine distances sequentially and applies EMA smoothing before normalization.

#### SU5 — Dynamic Time Warping

Uses Dynamic Time Warping (DTW) to evaluate the similarity between real and synthetic trajectories.

The implementation uses Haversine distance as the local distance metric and computes the average DTW distance along the alignment path.

Because of its computational cost, DTW was also evaluated separately from the real-time application.

---

### 2. Experimental Evaluation

Experiments were performed using trajectory datasets from:

* Porto
* San Francisco

The experiments evaluated different trajectory-length intervals and compared the implemented utility score methods.

The evaluation included:

* Mean Distance Error (MDE)
* Utility/accuracy scores
* Different trajectory length intervals
* Execution time
* Comparison between utility score methods

---

## 📊 Experimental Results

The experiments were performed using four trajectory-length intervals:

* 80–155 m
* 170–350 m
* 250–450 m
* 350–550 m

### SU1 — Original Method

| Trajectory length | MDE (km) | Accuracy |  Time |
| ----------------- | -------: | -------: | ----: |
| 80–155 m          |   0.1217 |   96.39% | ~40 s |
| 170–350 m         |   0.2363 |   96.50% | ~40 s |
| 250–450 m         |   0.3248 |   98.50% | ~40 s |
| 350–550 m         |   0.5778 |   60.30% | ~40 s |

### SU2 — Arithmetic Mean

| Trajectory length | MDE (km) | Accuracy |  Time |
| ----------------- | -------: | -------: | ----: |
| 80–155 m          |   0.1169 |   90.70% | ~41 s |
| 170–350 m         |   0.2616 |   80.70% | ~41 s |
| 250–450 m         |   0.3715 |   66.70% | ~41 s |
| 350–550 m         |   0.9857 |    9.60% | ~41 s |

### SU3 — Exponential Moving Average

| Trajectory length | MDE (km) | Accuracy |  Time |
| ----------------- | -------: | -------: | ----: |
| 80–155 m          |   0.1216 |   98.00% | ~42 s |
| 170–350 m         |   0.2557 |   99.50% | ~42 s |
| 250–450 m         |   0.3841 |   96.30% | ~42 s |
| 350–550 m         |   0.7484 |   20.70% | ~42 s |

### SU5 — Dynamic Time Warping

| Trajectory length | MDE (km) | Accuracy |  Time |
| ----------------- | -------: | -------: | ----: |
| 80–155 m          |   0.1230 |   88.40% | ~72 s |
| 170–350 m         |   0.2691 |   82.39% | ~73 s |
| 250–450 m         |   0.3550 |   69.80% | ~72 s |
| 350–550 m         |   1.0080 |    9.30% | ~73 s |

### Results Summary

The experiments indicate that:

* **SU3 (EMA)** achieved the best accuracy for the 80–155 m and 170–350 m intervals.
* **SU1** achieved the best results for the 250–450 m interval.
* SU3 generally provided a better balance between utility and computational cost than SU5.
* **SU5 (DTW)** required significantly more processing time.
* Performance decreased for all methods on the longest trajectory interval (350–550 m).

The experimental results are also visualized using the generated plots in the `plots/` directory.

---

# 📱 Mobile Application

A mobile application was developed to demonstrate the generation and visualization of synthetic GPS trajectories.

The application communicates with a Python backend through HTTP requests.

### Architecture

```text
Mobile Application
       │
       │ HTTP / JSON
       ▼
FastAPI Backend
       │
       ▼
APU-TrajGen+ Model
       │
       ▼
Synthetic Trajectory
       │
       ▼
OSRM Road Reconstruction
       │
       ▼
Mobile Map Visualization
```

### Technologies

**Frontend**

* React Native
* Expo
* TypeScript
* Expo Router

**Backend**

* Python
* FastAPI
* Uvicorn

**Trajectory Generation**

* GRU neural network
* APU-TrajGen+
* Adaptive trajectory generation

**Map/Road Reconstruction**

* OSRM

---

## 📱 Application Features

The application allows the user to:

* obtain GPS trajectory data;
* send trajectory information to the backend;
* generate synthetic trajectories;
* visualize the original and synthetic trajectories;
* reconstruct trajectories using road-network information;
* compare the generated trajectory with the original trajectory.

The mobile application source code is located in:

```text
frontend/
```

The FastAPI backend is implemented in:

```text
app.py
```

---

## 🗺️ Application Screenshots

### Original vs. Synthetic Trajectory

<!-- Add screenshot here -->

`//poza//`

### Mobile Application

<!-- Add screenshot here -->

`//poza//`

### Road Reconstruction using OSRM

<!-- Add screenshot here -->

`//poza//`

---

# 🧪 Experiments

The repository contains several Jupyter notebooks used during development and experimentation.

# 📊 Datasets

The experiments use trajectory data from the Porto and San Francisco datasets.

The original datasets are **not included in this repository** because of their size and dataset distribution considerations.

The local dataset directory is excluded using `.gitignore`:

```text
datasets/
```

Therefore, the experiments require the corresponding datasets to be obtained and prepared separately.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/D-Natalia/Apu-TrajgenPlus.git
cd Apu-TrajgenPlus
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Backend

The FastAPI backend can be started with:

```bash
uvicorn app:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

# 📱 Running the Mobile Application

Navigate to the frontend directory:

```bash
cd frontend
```

Install the JavaScript dependencies:

```bash
npm install
```

Start Expo:

```bash
npx expo start
```

The application can then be tested using an available Expo-compatible device or emulator.

The backend address must be configured appropriately for the device running the mobile application.

---

# Research Context

This project was developed as part of a bachelor's thesis focused on:

**Synthetic trajectory generation with privacy protection.**

The work investigates methods for evaluating the utility of synthetic GPS trajectories while preserving the privacy properties of the generated data.

The project combines:

* privacy-preserving synthetic data generation;
* recurrent neural networks;
* GPS trajectory processing;
* trajectory similarity metrics;
* utility evaluation;
* mobile application development;
* road-network reconstruction.

---

# 📚 Original Project and Publication

This work is based on and extends the original **APU-TrajGen+** project.

**Original repository:**
https://github.com/aromanrsc/APU-TrajGenPlus

The original project and its associated research publication should be cited when using the underlying APU-TrajGen+ framework.

<!-- Publication information can be added here. -->

---

# License

This repository contains an extension of the original APU-TrajGen+ project.

Please refer to the `LICENSE` file and the original project repository for information regarding licensing and redistribution.

---

##  Author

**Natalia-Noemi Demeter**

Bachelor's Degree in Computer Science

This repository documents the research, software development, experiments, and application development carried out as part of my bachelor's thesis.
