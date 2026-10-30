# -*- coding: utf-8 -*-
"""
WaterSip-DF-TTMC core functions.
"""
import os
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod   # older h5netcdf calls np.product, removed in numpy 2
import xarray as xr
import pandas as pd
import glob
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
import config


def open_nc(file_path):
    """
    Open a netCDF file, falling back to h5netcdf if the default engine is unavailable.
    """
    try:
        return xr.open_dataset(file_path)
    except (ValueError, ImportError, AttributeError, OSError):
        return xr.open_dataset(file_path, engine='h5netcdf')


# %% * * * * * * * * * * * * FLEXPART output to P simulation functions * * * * * * * * * * * *
def get_files(flexpart_output_file, start, end, time_span, key_string='partposit_'):
    start_date = datetime.strptime(start, '%Y%m%d%H')
    files = []
    while start_date <= datetime.strptime(end, '%Y%m%d%H'):
        file_str = start_date.strftime('%Y%m%d%H0000')
        file_name = f'{key_string}{file_str}'
        file_path = os.path.join(flexpart_output_file, file_name)
        files.append(file_path)
        start_date = start_date + timedelta(hours=time_span)
    return files


def get_and_combine_obs_files(observation_path, time_index, variable):
    result = []
    for t in time_index:
        pattern = os.path.join(observation_path, f'*{t}*.nc')
        ds = open_nc(glob.glob(pattern)[0])[variable]
        if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        result.append(ds)
    return xr.concat(result, dim='time')


def get_obs_step_sum(observation_path, times, variable, time_span):
    """Observed accumulation over (t, t+time_span] for each time t, aligned with partposit t -> t+time_span.
    ERA5 accumulations are stamped at the period end; the next month is also read to cover month-end steps."""
    span = pd.Timedelta(hours=time_span)
    months = times.union(times + span).strftime('%Y%m').drop_duplicates()
    ds = get_and_combine_obs_files(observation_path, months, variable)
    return ds.resample(time=f'{time_span}h', closed='right', label='left').sum(dim='time').sel(time=times)


def readpartposit_to_df(file, variables=None):

    variable_map = {
        'lon': 2,
        'lat': 3,
        'z': 4,
        'topo': 6,
        'q': 8,
        'dens': 9,
        'blh': 10,
        't': 12,
        'mass': 13
    }

    with open(file, 'rb') as f:
        nbytes = os.fstat(f.fileno()).st_size
        numpart = round((nbytes / 4 + 3) / 15 - 1)
        f.read(4 * 3)  # skip the head
        data_as_int = np.fromfile(f, dtype=np.int32, count=numpart * 15).reshape(-1, 15).T
        f.seek(4 * 3)  # reload position
        data_as_float = np.fromfile(f, dtype=np.float32, count=numpart * 15).reshape(-1, 15).T

    if variables is None:
        variables = list(variable_map.keys())

    output_data = {}
    for var in variables:
        if var in variable_map:
            output_data[var] = data_as_float[variable_map[var], :]
    output = pd.DataFrame(output_data, index=data_as_int[1, :])
    return output


def global_gridcell_info(spatial_resolution, lat_nor=90, lat_sou=-90, lon_lef=-180, lon_rig=180):
    Erad = 6.371e6  # [m] Earth radius
    latitude = np.arange(lat_nor, lat_sou - spatial_resolution, -spatial_resolution)
    longitude = np.arange(lon_lef, lon_rig + spatial_resolution, spatial_resolution)
    lat_n_bound = np.minimum(lat_nor, latitude + 0.5 * spatial_resolution)
    lat_s_bound = np.maximum(lat_sou, latitude - 0.5 * spatial_resolution)
    gridcell_area = np.zeros([len(latitude), 1])
    gridcell_area[:, 0] = (np.pi / 180.0) * Erad ** 2 * abs(
        np.sin(lat_s_bound * np.pi / 180.0) - np.sin(lat_n_bound * np.pi / 180.0)) * spatial_resolution  # m^2
    gridcell_area = np.tile(gridcell_area, len(longitude))
    return latitude, longitude, gridcell_area


def write_to_nc_3d(data, var_name, file_name, time, latitude, longitude, output_path):
    result = xr.DataArray(data,
                          coords={"time": time, "latitude": latitude, "longitude": longitude},
                          dims=["time", "latitude", "longitude"],
                          name=var_name)
    result.to_netcdf(os.path.join(output_path, "{}.nc".format(file_name)))
    print(file_name, 'write to nc done.')


def write_to_nc_2d(data, var_name, file_name, latitude, longitude, output_path):
    result = xr.DataArray(data,
                          coords={"latitude": latitude, "longitude": longitude},
                          dims=["latitude", "longitude"],
                          name=var_name)
    result.to_netcdf(os.path.join(output_path, "{}.nc".format(file_name)))
    print(file_name, 'write to nc done.')


def calculate_RH(df, RH_name='RH', dens='dens', q='q', t='t'):

    p = df[dens] * 2.8705 * (1 + 0.608 * df[q]) * df[t]  # hpa, P=ρ⋅Rd(1+0.608q)⋅T
    df[RH_name] = 26.3 * p * df[q] / np.exp((17.67 * (df[t] - 273.16)) / (df[t] - 29.65))


def midpoint(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)

    x, y, z = (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2

    mid_lon = np.arctan2(y, x)
    mid_lat = np.arctan2(z, np.sqrt(x**2 + y**2))

    return np.degrees(mid_lat), np.degrees(mid_lon)


def read_shp(shp_file):
    """
    Read a shapefile, preferring the pyogrio engine and falling back to the default.
    """
    try:
        return gpd.read_file(shp_file, engine='pyogrio')
    except Exception:
        return gpd.read_file(shp_file)


def from_bounds_get_parts(df, lat_up, lat_down, lon_left, lon_right):
    """
    Select particles inside a lat/lon box.
    """
    return df[(df['lat'] >= lat_down) & (df['lat'] <= lat_up) &
              (df['lon'] >= lon_left) & (df['lon'] <= lon_right)]


def make_region_filter(region):
    """
    Build a reusable filter df -> df[inside target region].
    region: shapefile path, [lat_up, lat_down, lon_left, lon_right], or anything else (no filtering).
    Shapefiles are read once and tested with vectorized shapely.contains_xy.
    """
    if isinstance(region, str) and region.endswith('.shp'):
        import shapely
        gdf = read_shp(region)
        geometry = gdf.union_all() if hasattr(gdf, 'union_all') else gdf.unary_union
        shapely.prepare(geometry)

        def _filter(df):
            mask = shapely.contains_xy(
                geometry,
                np.asarray(df['lon'].values, dtype=np.float64),
                np.asarray(df['lat'].values, dtype=np.float64))
            return df[mask]
        return _filter
    if isinstance(region, list):
        return lambda df: from_bounds_get_parts(df, region[0], region[1],
                                                region[2], region[3])
    return lambda df: df


def back_tracking_files(flexpart_output_file, start_time, tracking_days, time_span, key_string='partposit_'):
    """
    List partposit files for backward tracking.
    """
    start_date = datetime.strptime(start_time, '%Y%m%d%H')
    end_date = start_date - timedelta(days=tracking_days)
    files = []
    while start_date >= end_date:
        file_str = start_date.strftime('%Y%m%d%H0000')
        file_name = f'{key_string}{file_str}'
        file_path = os.path.join(flexpart_output_file, file_name)
        files.append(file_path)
        start_date = start_date - timedelta(hours=time_span)
    return files


def from_shp_get_mask(shp, latitude, longitude):
    """
    Grid mask from a shapefile.
    """
    gdf = read_shp(shp)
    geometry = gdf.unary_union

    mask = np.zeros((len(latitude), len(longitude)), dtype=bool)
    for i, lat in enumerate(latitude):
        for j, lon in enumerate(longitude):
            point = Point(lon, lat)
            mask[i, j] = geometry.contains(point)
    return mask


def from_bounds_get_mask(lat_up, lat_down, lon_left, lon_right, latitude, longitude):
    """
    Grid mask from a lat/lon box.
    """
    lat_mask = (latitude >= lat_down) & (latitude <= lat_up)
    lon_mask = (longitude >= lon_left) & (longitude <= lon_right)
    return np.outer(lat_mask, lon_mask)


def calculate_coordinate_round(coordinate_value, output_spatial_resolution):
    """
    Round coordinates to the output grid.
    """
    return np.round(coordinate_value / output_spatial_resolution) * output_spatial_resolution


def get_algorithm_params(method):
    """
    Get parameters for the given method.
    """
    if method == 'WaterSip':
        return {
            'q_diff_p': config.WATERSIP_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_Q_DIFF_E,
            'rh_threshold': config.WATERSIP_RH_THRESHOLD,
            'blh_factor': config.WATERSIP_BLH_FACTOR,
            'update_temporary': config.WATERSIP_UPDATE_TEMPORARY
        }
    elif method == 'WaterSip-DF':
        return {
            'q_diff_p': config.WATERSIP_DF_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_DF_Q_DIFF_E,
            'update_df': config.WATERSIP_DF_UPDATE_DF,
            'update_temporary': config.WATERSIP_DF_UPDATE_TEMPORARY
        }
    elif method == 'WaterSip-TTMC':
        return {
            'q_diff_p': config.WATERSIP_TTMC_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_TTMC_Q_DIFF_E,
            'rh_threshold': config.WATERSIP_TTMC_RH_THRESHOLD,
            'blh_factor': config.WATERSIP_TTMC_BLH_FACTOR,
            'update_p_simulation': config.WATERSIP_TTMC_UPDATE_P_SIMULATION,
            'update_temporary': config.WATERSIP_TTMC_UPDATE_TEMPORARY
        }
    elif method == 'WaterSip-DF-TTMC':
        return {
            'q_diff_p': config.WATERSIP_DF_TTMC_Q_DIFF_P,
            'q_diff_e': config.WATERSIP_DF_TTMC_Q_DIFF_E,
            'update_df': config.WATERSIP_DF_TTMC_UPDATE_DF,
            'update_p_simulation': config.WATERSIP_DF_TTMC_UPDATE_P_SIMULATION,
            'update_temporary': config.WATERSIP_DF_TTMC_UPDATE_TEMPORARY
        }
    else:
        raise ValueError(f"Unknown method: {method}")
