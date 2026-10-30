# -*- coding: utf-8 -*-
"""
Step 1: Dynamic RH thresholds calculation
"""
import os
import pandas as pd
import numpy as np
import threading
from functions import (get_obs_step_sum, get_files, readpartposit_to_df,
                      global_gridcell_info, calculate_RH, midpoint, write_to_nc_2d,
                      calculate_coordinate_round, get_algorithm_params)
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

MAX_WORKERS = min(4, os.cpu_count())
file_write_lock = threading.Lock()  # serialize NetCDF writes across threads


def dynamic_rh_thresholds_calculation(method, skip_existing=False):
    """Dynamic RH thresholds. skip_existing=True skips times whose output already exists."""
    print(f'\nStep 1 ({method}): Calculate dynamic RH thresholds...')

    params = get_algorithm_params(method)
    os.makedirs(config.DF_FILE_PATH, exist_ok=True)

    time = pd.date_range(start=pd.to_datetime(config.START_TIME, format='%Y%m%d%H'),
                         end=pd.to_datetime(config.END_TIME, format='%Y%m%d%H'),
                         freq='{}h'.format(config.TIME_SPAN))[:-1]

    P_era5_6h = get_obs_step_sum(config.OBSERVATION_PATH, time, 'tp', config.TIME_SPAN) * 1000

    files = get_files(config.PARTPOSIT_PATH, config.START_TIME, config.END_TIME, config.TIME_SPAN)

    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION,
                                                               lat_nor=90, lat_sou=-90,
                                                               lon_lef=-179, lon_rig=180)

    P_era5_6h = P_era5_6h.where(P_era5_6h > 0, 0)
    P_era5_6h = P_era5_6h * gridcell_area

    def generate_files(i, files, latitude, longitude, P_era5_6h, time):
        if skip_existing and os.path.exists(os.path.join(
                config.DF_FILE_PATH,
                'DF_RH_thresholds_{}.nc'.format(time[i - 1].strftime('%Y%m%d%H')))):
            return
        df = readpartposit_to_df(files[i], variables=['lat', 'lon', 'q', 't', 'dens', 'mass'])
        calculate_RH(df)
        df = df.drop(columns=['t', 'dens'])

        df0 = readpartposit_to_df(files[i - 1], variables=['lat', 'lon', 'q', 't', 'dens'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
        calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
        df0 = df0.drop(columns=['t0', 'dens0'])

        df = df.merge(df0, left_index=True, right_index=True)

        df['q_diff'] = df['q'] - df['q0']
        df = df[df['q_diff'] < params['q_diff_p']]
        df = df.drop(columns=['q', 'q0'])

        df['lat'], df['lon'] = midpoint(df['lat'].values, df['lon'].values,
                                        df['lat0'].values, df['lon0'].values)
        df = df.drop(columns=['lat0', 'lon0'])

        df['RH'] = (df['RH'] + df['RH0']) / 2
        df = df.drop(columns=['RH0'])

        df['lat_round'] = calculate_coordinate_round(df['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
        df['lon_round'] = calculate_coordinate_round(df['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
        df = df.drop(columns=['lon', 'lat'])

        RH_span = np.arange(0, 120, 0.25)
        p_simulation = np.zeros([len(RH_span), len(latitude), len(longitude)])

        for j in range(0, len(RH_span)):
            df0 = df.copy()
            df0 = df0[df0['RH'] > RH_span[j]]
            df0['p_mass'] = -df0['mass'] * df0['q_diff']
            df_grouped_p = df0.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()
            # sum P on grids
            lat_idx_p = ((latitude[0] - df_grouped_p['lat_round']) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            # np.add.at: lon_round -180 and 180 map to the same column; plain assignment would drop mass
            np.add.at(p_simulation[j], (lat_idx_p.values, lon_idx_p.values),
                      df_grouped_p['p_mass'].values)

        obs0 = P_era5_6h[i - 1].values
        min_indices = np.nanargmin(np.abs(p_simulation - obs0), axis=0)

        DF_RH_thresholds = np.where(np.all(p_simulation == 0, axis=0), np.nan, RH_span[min_indices])
        DF_RH_thresholds[obs0 == 0] = 120  # sentinel (beyond RH_span) for cells without observed precipitation

        with file_write_lock:
            write_to_nc_2d(DF_RH_thresholds, 'DF_RH_thresholds',
                           'DF_RH_thresholds_{}'.format(time[i - 1].strftime('%Y%m%d%H')),
                           latitude, longitude, config.DF_FILE_PATH)

        del df, df0, p_simulation, obs0, DF_RH_thresholds

    print(f'Processing {len(files)-1} times with {MAX_WORKERS} workers...')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(generate_files, i, files, latitude, longitude, P_era5_6h, time): i
            for i in range(1, len(files))
        }

        n_failed = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                future.result()
            except Exception as e:
                # a failed time step (e.g. corrupted partposit) does not stop the run; rerun with skip_existing
                n_failed += 1
                print(f'! Step 1 failed for time index {i} '
                      f'({time[i - 1].strftime("%Y%m%d%H")}): {e}')
        if n_failed:
            print(f'! Step 1: {n_failed} time(s) failed, see messages above')

    print('Step 1 done!')


if __name__ == "__main__":
    dynamic_rh_thresholds_calculation('WaterSip-DF-TTMC')
