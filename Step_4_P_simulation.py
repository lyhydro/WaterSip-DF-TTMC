# -*- coding: utf-8 -*-
"""
Step 4: P simulation (optional; global simulated precipitation for evaluation)
"""
import os
import pandas as pd
import numpy as np
from functions import (get_files, readpartposit_to_df, global_gridcell_info,
                      calculate_RH, midpoint, write_to_nc_2d, open_nc,
                      calculate_coordinate_round, get_algorithm_params)
import config


def p_simulation(method, df_method, skip_existing=False):
    """P simulation. skip_existing=True skips times whose output already exists."""
    print(f'Step 4 ({method}): P simulation...')

    params = get_algorithm_params(method)
    os.makedirs(config.P_E_SIMULATION_OUTPUT_PATH, exist_ok=True)
    files = get_files(config.PARTPOSIT_PATH, config.START_TIME, config.END_TIME, config.TIME_SPAN)

    latitude, longitude, gridcell_area = global_gridcell_info(config.OUTPUT_SPATIAL_RESOLUTION,
                                                              lat_nor=90, lat_sou=-90,
                                                              lon_lef=-179, lon_rig=180)

    temp_time = pd.date_range(start=pd.to_datetime(config.START_TIME, format='%Y%m%d%H'),
                              end=pd.to_datetime(config.END_TIME, format='%Y%m%d%H'),
                              freq='{}h'.format(config.TIME_SPAN))[:-1]

    n_failed = 0
    for i in range(1, len(files)):  # from 1 start, the files[0] used to calculate diff
        if skip_existing and os.path.exists(os.path.join(
                config.P_E_SIMULATION_OUTPUT_PATH,
                f'P_simulation_mm_{temp_time[i - 1].strftime("%Y%m%d%H")}.nc')):
            continue
        try:
            _p_simulation_one(i, files, temp_time, params, df_method,
                              latitude, longitude, gridcell_area)
        except Exception as e:
            # a failed time step (e.g. corrupted partposit) does not stop the run
            n_failed += 1
            print(f'! Step 4 failed for {temp_time[i - 1].strftime("%Y%m%d%H")}: {e}')
    if n_failed:
        print(f'! Step 4: {n_failed} time(s) failed, see messages above')

    print('Step 4 done!')


def _p_simulation_one(i, files, temp_time, params, df_method,
                      latitude, longitude, gridcell_area):
    """P simulation for a single time step."""
    if True:
        df_p = readpartposit_to_df(files[i], variables=['lon', 'lat', 'q', 't', 'dens', 'mass'])
        calculate_RH(df_p)
        df_p = df_p.drop(columns=['dens', 't'])

        df0 = readpartposit_to_df(files[i - 1], variables=['lon', 'lat', 'q', 't', 'dens'])
        df0 = df0.rename(columns={'lat': 'lat0', 'lon': 'lon0', 'q': 'q0', 't': 't0', 'dens': 'dens0'})
        calculate_RH(df0, RH_name='RH0', dens='dens0', q='q0', t='t0')
        df0 = df0.drop(columns=['dens0', 't0'])

        df_p = df_p.merge(df0, left_index=True, right_index=True)

        df_p['q_diff'] = df_p['q'] - df_p['q0']
        df_p = df_p.drop(columns=['q', 'q0'])
        df_p = df_p[df_p['q_diff'] < 0]  # pre-filter

        df_p['RH'] = (df_p['RH'] + df_p['RH0']) / 2
        df_p = df_p.drop(columns=['RH0'])

        df_p['lat'], df_p['lon'] = midpoint(df_p['lat'].values, df_p['lon'].values,
                                            df_p['lat0'].values, df_p['lon0'].values)
        df_p = df_p.drop(columns=['lat0', 'lon0'])

        df_p['lat_round'] = calculate_coordinate_round(df_p['lat'], config.OUTPUT_SPATIAL_RESOLUTION)
        df_p['lon_round'] = calculate_coordinate_round(df_p['lon'], config.OUTPUT_SPATIAL_RESOLUTION)
        df_p = df_p.drop(columns=['lat', 'lon'])

        # RH filter: dynamic thresholds (DF) or fixed threshold
        if df_method == 'on':
            RH_threshold = open_nc(
                f'{config.DF_FILE_PATH}/DF_RH_thresholds_{temp_time[i - 1].strftime("%Y%m%d%H")}.nc'
            )['DF_RH_thresholds']
            df_RH_threshold = RH_threshold.to_dataframe(name='RH_threshold').reset_index()
            df_RH_threshold = df_RH_threshold.query("RH_threshold != 100 and RH_threshold.notna()", engine='numexpr')

            df_p = pd.merge(df_p, df_RH_threshold, how='left',
                           left_on=['lat_round', 'lon_round'],
                           right_on=['latitude', 'longitude'])
            df_p = df_p.drop(columns=['latitude', 'longitude'])
            df_p.dropna(subset=['RH_threshold'], inplace=True)
            df_p = df_p[df_p['RH'] >= df_p['RH_threshold']]
        else:
            df_p = df_p[df_p['q_diff'] < params['q_diff_p']]
            df_p = df_p[df_p['RH'] >= params['rh_threshold']]

        df_p['p_mass'] = -df_p['mass'] * df_p['q_diff']  # kg, *-1 makes P positive
        df_grouped_p = df_p.groupby(['lat_round', 'lon_round'])['p_mass'].sum().reset_index()

        P_simulation_mm = np.zeros([len(latitude), len(longitude)])
        if len(df_grouped_p) > 0:
            lat_idx_p = ((latitude[0] - df_grouped_p['lat_round']) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            lon_idx_p = ((df_grouped_p['lon_round'] - longitude[0]) / config.OUTPUT_SPATIAL_RESOLUTION).astype(int)
            # np.add.at: lon_round -180 and 180 map to the same column; plain assignment would drop mass
            np.add.at(P_simulation_mm, (lat_idx_p.values, lon_idx_p.values),
                      df_grouped_p['p_mass'].values)

        P_simulation_mm = P_simulation_mm / gridcell_area  # kg/m^2 = mm

        write_to_nc_2d(P_simulation_mm, 'P_simulation_mm',
                       f'P_simulation_mm_{temp_time[i - 1].strftime("%Y%m%d%H")}',
                       latitude, longitude, config.P_E_SIMULATION_OUTPUT_PATH)

        del df_p, df0, df_grouped_p, P_simulation_mm


if __name__ == "__main__":
    p_simulation('WaterSip-DF-TTMC', 'on')  # 'WaterSip', 'off' ; 'WaterSip-TTMC', 'off' ; 'WaterSip-DF-TTMC', 'on'
