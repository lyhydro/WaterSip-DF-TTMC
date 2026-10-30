# -*- coding: utf-8 -*-
"""
Step 5: Moisture contribution calculation (with TTMC correction)

Input: Step 3 trajectory files (*_P_particle_trajectories.nc), dims (particle, time).
"""
import os
import re
import numpy as np
import pandas as pd
from datetime import timedelta
from functions import (get_obs_step_sum, global_gridcell_info,
                       from_shp_get_mask, from_bounds_get_mask, write_to_nc_3d, open_nc)
import config


def _get_traj_files(traj_path, start, end, time_span):
    """List trajectory files in [start, end), sorted by time."""
    pat = re.compile(r'(\d{10})-\d+_P_particle_trajectories\.nc$')
    t_start = pd.to_datetime(start, format='%Y%m%d%H')
    t_end = pd.to_datetime(end, format='%Y%m%d%H') - timedelta(hours=time_span)
    files = []
    for f in sorted(os.listdir(traj_path)):
        m = pat.match(f)
        if not m:
            continue
        t = pd.to_datetime(m.group(1), format='%Y%m%d%H')
        if t_start <= t <= t_end:
            files.append((t, os.path.join(traj_path, f)))
    return files


def _contribution_from_nc(ds, latitude, longitude, res):
    """
    Gridded moisture contribution for one arrival time.
    Returns (T, nlat, nlon); T = tracking steps including step 0 (arrival).
    """
    if len(ds['particle']) == 0:
        T = len(ds['time'])
        return np.zeros((T, len(latitude), len(longitude)), dtype=np.float64)

    # (particle, time) -> (time, particle)
    lon_t = ds['lon'].values.T    # (T, N)
    lat_t = ds['lat'].values.T
    f_t = ds['f'].values.T        # (T, N)
    p_mass = ds['p_mass'].values  # (N,)

    T, N = lon_t.shape
    nlat, nlon = len(latitude), len(longitude)
    out = np.zeros((T, nlat, nlon), dtype=np.float64)

    # nearest grid indices
    lat_idx = np.argmin(np.abs(latitude[:, None] - lat_t.reshape(-1)), axis=0).reshape(T, N)
    lon_idx = np.argmin(np.abs(longitude[:, None] - lon_t.reshape(-1)), axis=0).reshape(T, N)

    for j in range(1, T):
        fj = f_t[j]                          # (N,)
        valid = ~(np.isnan(lat_t[j]) | np.isnan(lon_t[j]) | np.isnan(fj) | (fj == 0))
        if not valid.any():
            continue
        np.add.at(out[j],
                  (lat_idx[j, valid], lon_idx[j, valid]),
                  p_mass[valid] * fj[valid])
    return out


def calculate_moisture_contribution(df_method, ttmc_method):
    """Moisture contribution, with TTMC correction if ttmc_method == 'on'."""
    print('\nStep 5: Calculate moisture contribution...')

    latitude, longitude, gridcell_area = global_gridcell_info(
        config.OUTPUT_SPATIAL_RESOLUTION,
        lat_nor=90, lat_sou=-90, lon_lef=-179, lon_rig=180)

    traj_files = _get_traj_files(config.TEMPORARY_FILE_PATH,
                                 config.START_TIME, config.END_TIME, config.TIME_SPAN)
    time_index = pd.DatetimeIndex([t for t, _ in traj_files])

    if ttmc_method == 'on':
        print("Loading obs data...")
        obs_p = get_obs_step_sum(config.OBSERVATION_PATH, time_index, 'tp', config.TIME_SPAN) * 1000
        obs_p = obs_p.where(obs_p > 0, 0)

        if isinstance(config.TARGET_REGION, str) and config.TARGET_REGION.endswith('.shp'):
            mask = from_shp_get_mask(config.TARGET_REGION, latitude, longitude)
        elif isinstance(config.TARGET_REGION, list):
            mask = from_bounds_get_mask(config.TARGET_REGION[0], config.TARGET_REGION[1],
                                        config.TARGET_REGION[2], config.TARGET_REGION[3],
                                        latitude, longitude)
        else:
            mask = np.ones((len(latitude), len(longitude)))

    moisture_contribution_mm = np.zeros([len(traj_files), len(latitude), len(longitude)])

    for n, (time_p, fpath) in enumerate(traj_files):
        ds = open_nc(fpath)
        if len(ds['particle']) == 0:
            print(time_p.strftime('%Y%m%d%H'), 'no particles, skip')
            ds.close()
            continue

        contrib_t = _contribution_from_nc(ds, latitude, longitude,
                                          config.OUTPUT_SPATIAL_RESOLUTION)
        ds.close()
        contrib_t = contrib_t[::-1] / gridcell_area

        if ttmc_method == 'on':
            # TTMC: cc_p = observed target-region precipitation / total tracked moisture,
            # so the corrected contributions sum to the observed precipitation
            obs_p_selected = obs_p.sel(time=time_p)
            obs_total = np.sum(obs_p_selected.values * gridcell_area * mask)
            tracked_total = np.sum(np.sum(contrib_t, axis=0) * gridcell_area)
            with np.errstate(divide='ignore', invalid='ignore'):
                cc_p = obs_total / tracked_total
            if not np.isfinite(cc_p):
                cc_p = 1.0
            moisture_contribution_mm[n] = np.sum(contrib_t, axis=0) * cc_p
        else:
            moisture_contribution_mm[n] = np.sum(contrib_t, axis=0)

        print(time_p.strftime('%Y%m%d%H'), 'done')

    if df_method == 'on' and ttmc_method == 'on':
        output_name_mm = f'moisture_contribution_mm_DF_TTMC_{config.START_TIME}_{config.END_TIME}'
    elif df_method == 'on' and ttmc_method == 'off':
        output_name_mm = f'moisture_contribution_mm_DF_{config.START_TIME}_{config.END_TIME}'
    elif df_method == 'off' and ttmc_method == 'on':
        output_name_mm = f'moisture_contribution_mm_TTMC_{config.START_TIME}_{config.END_TIME}'
    else:
        output_name_mm = f'moisture_contribution_mm_{config.START_TIME}_{config.END_TIME}'

    os.makedirs(config.FINAL_OUTPUT_PATH, exist_ok=True)
    write_to_nc_3d(moisture_contribution_mm, 'moisture_contribution_mm', output_name_mm,
                   time_index, latitude, longitude, config.FINAL_OUTPUT_PATH)
    print('Step 5 done!')


if __name__ == "__main__":
    calculate_moisture_contribution(df_method='on', ttmc_method='on')
