# APU-TrajGen+

This is the code for generating the results presented in the article: ***APU-TrajGen+: GRU-based Adaptive Privacy and
Utility Preserving Trajectory Generation***

## Environment Requirements

- Python >= 3.11.5
- The requirements.txt file is included

## Datasets

- Porto - https://doi.org/10.24432/C55W25
- San Francisco - https://ieee-dataport.org/open-access/crawdad-epflmobility

## Reproducibility & Run

### Model generation

=> "training" folder

### Trajectory generation

=> "trajgen" folder

**Step 0**. Generate the test data files using the [trajgen-generate-data](trajgen/trajgen-generate-data.ipynb) notebook, for the targeted dataset.

**Step 1**. Run the [trajgen-fixed-k](trajgen/trajgen-fixed-k.ipynb) notebook for the targeted dataset.

**Step 2**. Extract the values of **MDE_k** for various values of k (k >= 1). MDE_k values are necessary for setting up the adaptive k approach.

**Step 3**. Run the [trajgen-adaptive-k](trajgen/trajgen-adaptive-k.ipynb) notebook for generating the trajectories using the adaptive k approach.

**Step 4**. Compute **privacy and utility metrics** using the [trajgen-metrics-computation](trajgen/trajgen-metrics-computation.ipynb) notebook.

**Step 5**. **Attack resilience**: simulate the reconctruction of real trajectories from the published synthetic
trajectories ([trajgen-reconstruction](trajgen/trajgen-reconstruction.ipynb)) and compute the **Reconstruction Error (RE)** ([trajgen-reconstruction-error](trajgen/trajgen-reconstruction-error.ipynb)).

### Use case

(to be added)

### Results

<p float="left">
<img src="./trajgen/output/aputraj_porto_adaptive_k_mean_min80_mean_max155_trajid_5.png" width="49%" />
<img src="./trajgen/output/aputraj_porto_adaptive_k_mean_min170_mean_max350_trajid_5.png" width="49%" />
Example of a trajectory generation from the San Francisco dataset using APU-TrajGen+ with adaptive k
</p>

<p float="left">
<img src="./trajgen/output/aputraj_sanfrancisco_adaptive_k_mean_min170_mean_max350_trajid_8.png" width="49%" />
<img src="./trajgen/output/aputraj_sanfrancisco_adaptive_k_mean_min250_mean_max450_trajid_8.png" width="49%" />
Example of a trajectory generation from the San Francisco dataset using APU-TrajGen+ with adaptive k
</p>


## Acknowledgement

Any scientific publications that use our data or code should mention the following article:

```
@ARTICLE{11162526,
  author={Roman, Adrian-Silviu and Bolboacă, Roland and Lenard, Teri and Haller, Piroska},
  journal={IEEE Access}, 
  title={APU-TrajGen+: GRU-Based Adaptive Privacy and Utility Preserving Trajectory Generation}, 
  year={2025},
  volume={13},
  number={},
  pages={160081-160102},
  keywords={Trajectory;Privacy;Data privacy;Data models;Protection;Real-time systems;Location data privacy;trajectory protection;trajectory generation;real-time trajectory release;data utility;traffic management systems},
  doi={10.1109/ACCESS.2025.3609406}}
```
