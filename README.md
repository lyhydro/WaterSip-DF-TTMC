This is the Python project that can be used to implement the WaterSip (Sodemann et al., 2008), WaterSip-DF, WaterSip-TTMC, and WaterSip-DF-TTMC methods. It is the updated version of [WaterSip-DF](https://github.com/lyhydro/WaterSip-DF). For detailed information on the DF methods, please refer to "Dynamic Filtering in Lagrangian Model-Based Moisture Source-Receptor Diagnosis" (https://doi.org/10.1029/2026JD047402).

**DF** (Dynamic Filtering) uses observed precipitation/evaporation (e.g. ERA5) to derive dynamic RH thresholds and BLH scaling factors for every grid cell and time step, replacing the fixed thresholds of WaterSip.

**TTMC** (Total Tracked Moisture Correction) replaces the HAMSTER bias correction used in WaterSip-DF. For each time step, a single factor `cc_p = observed precipitation over the target region / total tracked moisture` scales the whole moisture contribution field, so that the corrected contributions sum to the observed precipitation. 

The project consists of five core scripts (Step 1–5):

| Method | Steps |
|---|---|
| WaterSip | Step 3 → 5 |
| WaterSip-DF | Step 1 → 2 → 3 → 5 |
| WaterSip-TTMC | Step 3 → 5 |
| WaterSip-DF-TTMC | Step 1 → 2 → 3 → 5 |

- Step 1: dynamic RH thresholds (DF)
- Step 2: dynamic BLH factors (DF)
- Step 3: particle filtering and backward tracking
- Step 4: global precipitation simulation (optional, for evaluation only)
- Step 5: moisture contribution calculation, with TTMC correction for the TTMC methods

Users can define target regions using either rectangular coordinates or shapefiles (.shp). Intermediate particle trajectories are saved as NetCDF files for further diagnostics.

#### Install
1. Clone the repository
```shell
git clone https://github.com/lyhydro/WaterSip-DF-TTMC
cd WaterSip-DF-TTMC
```
2. Make an environment with the necessary python packages
```shell
pip install -r requirements.txt
```

#### Configuration
Refer to and modify the **config.py** file. Set the `UPDATE_*` switches to `True` for the first run; set them to `False` later to reuse existing intermediate results.

#### Required data
1. "partposit*" files (FLEXPART output under domain-filling mode), covering `START_TIME - TRACKING_DAYS` to `END_TIME`. Default folder: **partposit_file**.
2. For the DF and TTMC methods, ERA5 hourly precipitation (`tp`) and evaporation (`e`) in monthly .nc files (file names containing `YYYYMM`), including one extra month after `END_TIME`. Default folder: **P_E_observation**.
3. If using a shapefile to define the target region, place the .shp file into the **shp** folder.

#### Run the methods
```shell
# WaterSip
python main.py watersip
# WaterSip-DF
python main.py watersip-df
# WaterSip-TTMC
python main.py watersip-ttmc
# WaterSip-DF-TTMC
python main.py watersip-df-ttmc
```

#### Output
`moisture_contribution_mm_*.nc` (time × latitude × longitude) in the **output** folder: the moisture contribution (mm) of each grid cell to precipitation over the target region at each time step. 
