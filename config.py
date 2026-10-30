# -*- coding: utf-8 -*-
"""
WaterSip-DF-TTMC configuration
"""

# ==================== PATH ====================
PARTPOSIT_PATH = r'./partposit_file' # FLEXPART output (partposit*)
TEMPORARY_FILE_PATH = r'./temporary' # Step 3 trajectory files (.nc)
FINAL_OUTPUT_PATH = r'./output' # Store final output (.nc)
# useful for DF method (WaterSip-DF & WaterSip-DF-TTMC)
DF_FILE_PATH = r'./DF' # Step 1/2 dynamic RH thresholds and BLH factors
# useful for DF and TTMC methods
OBSERVATION_PATH = r'./P_E_observation' # currently only support ERA5
P_E_SIMULATION_OUTPUT_PATH = r'./P_E_simulation' # Step 4 P simulation (optional)

# ==================== Basic Configuration ====================
START_TIME = '2024010100' # 'YearMonthDayHour'
END_TIME = '2024020100' # 'YearMonthDayHour', not include time_span after the time
TRACKING_DAYS = 20 # days
TIME_SPAN = 6 # Time span of FLEXPART output (partposit*)
OUTPUT_SPATIAL_RESOLUTION = 1 # Degree of output grids
TARGET_REGION = r'./shp/boundary.shp'  # r'./shp/boundary.shp' or [lat_up, lat_down, lon_left, lon_right]

# ==================== WaterSip ====================
WATERSIP_Q_DIFF_P = -0.0000  # kg/kg
WATERSIP_Q_DIFF_E = 0.0002 # kg/kg
WATERSIP_RH_THRESHOLD = 80 # %
WATERSIP_BLH_FACTOR = 1.5 # Scaling factor to BLH
WATERSIP_UPDATE_TEMPORARY = True # True or False

# ==================== WaterSip-DF ====================
WATERSIP_DF_Q_DIFF_P = -0.0000 # kg/kg
WATERSIP_DF_Q_DIFF_E = 0.0000 # kg/kg
WATERSIP_DF_UPDATE_DF = True # True or False
WATERSIP_DF_UPDATE_TEMPORARY = True # True or False

# ==================== WaterSip-TTMC ====================
WATERSIP_TTMC_Q_DIFF_P = -0.0000 # kg/kg
WATERSIP_TTMC_Q_DIFF_E = 0.0002 # kg/kg
WATERSIP_TTMC_RH_THRESHOLD = 80 # %
WATERSIP_TTMC_BLH_FACTOR = 1.5 # Scaling factor to BLH
WATERSIP_TTMC_UPDATE_P_SIMULATION = False # True or False, Step 4 (optional, not required by TTMC)
WATERSIP_TTMC_UPDATE_TEMPORARY = True # True or False

# ==================== WaterSip-DF-TTMC ====================
WATERSIP_DF_TTMC_Q_DIFF_P = -0.0000 # kg/kg
WATERSIP_DF_TTMC_Q_DIFF_E = 0.0000 # kg/kg
WATERSIP_DF_TTMC_UPDATE_DF = True # True or False, update Step 1/2 (DF thresholds)
WATERSIP_DF_TTMC_UPDATE_TEMPORARY = True # True or False, update Step 3 (trajectories)
WATERSIP_DF_TTMC_UPDATE_P_SIMULATION = False # True or False, Step 4 (optional, not required by TTMC)
