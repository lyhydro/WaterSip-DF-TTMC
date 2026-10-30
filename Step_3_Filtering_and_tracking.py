# -*- coding: utf-8 -*-
"""
Step 3: particle filtering and backward moisture tracking.

- Scans partposit files in descending time order; each file is read once and
  serves all active tracking windows.
- Writes one NetCDF per precipitation time, {times}-{span}_P_particle_trajectories.nc
  (dims particle x time, zlib-compressed).
"""
import os
import gc
import time
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from functions import (readpartposit_to_df, calculate_RH, midpoint,
                       open_nc, calculate_coordinate_round, get_algorithm_params,
                       make_region_filter)
import config

_SLICE_COLS = ['lon', 'lat', 'z', 'q', 'blh', 'mass', 't', 'topo']


def _prefilter_particles(df, df0, tmp_time, params):
    """
    Region-independent prefilter (q_diff threshold + midpoints), shared by all regions.
    df: partposit at tmp_time + TIME_SPAN; df0: partposit at tmp_time (both with RH).
    Returns lat/lon (midpoint), q_diff, RH, mass; None if no particle passes q_diff.
    """
    common_indices = df.index.intersection(df0.index)
    df = df.loc[common_indices]
    df0 = df0.loc[common_indices]

    q_diff = df['q'].values - df0['q'].values          # float32
    RH_avg = (df['RH'].values + df0['RH'].values) / 2  # float32
    result_df = pd.DataFrame({
        'lat': df['lat'].values, 'lon': df['lon'].values,
        'lat_0': df0['lat'].values, 'lon_0': df0['lon'].values,
        'q_diff': q_diff, 'RH': RH_avg, 'mass': df['mass'].values,
    }, index=df.index)
    result_df.index.name = 'index'
    result_df = result_df[result_df['q_diff'].values < params['q_diff_p']]

    if len(result_df) == 0:
        print(f"No particles after q_diff filtering for {tmp_time}")
        return None

    result_df['lat'], result_df['lon'] = midpoint(result_df['lat'].values,
                                                  result_df['lon'].values,
                                                  result_df['lat_0'].values,
                                                  result_df['lon_0'].values)
    result_df = result_df.drop(columns=['lat_0', 'lon_0'])
    return result_df


def _rh_filter_and_pmass(result_df, tmp_time, params, df_method, rh_threshold_loader):
    """Apply the RH threshold (dynamic or fixed) and compute p_mass."""
    if df_method == 'on':
        # Dynamic RH threshold per grid cell; keeps row order
        lat_round = calculate_coordinate_round(result_df['lat'].values,
                                               config.OUTPUT_SPATIAL_RESOLUTION).astype(np.float64)
        lon_round = calculate_coordinate_round(result_df['lon'].values,
                                               config.OUTPUT_SPATIAL_RESOLUTION).astype(np.float64)
        thr_lookup = rh_threshold_loader(tmp_time)
        keys = pd.MultiIndex.from_arrays([lat_round, lon_round])
        thr_vals = thr_lookup.reindex(keys).values      # float64, NaN where missing
        keep = result_df['RH'].values >= thr_vals       # NaN compares False -> dropped
        result_df = result_df[keep]
        result_df = result_df[['lat', 'lon', 'q_diff', 'mass']]
    else:
        rh_threshold = params.get('rh_threshold')
        result_df = result_df[result_df['RH'] > rh_threshold]
        result_df = result_df.drop(columns=['RH'])

    result_df['p_mass'] = -result_df['mass'] * result_df['q_diff']
    return result_df


def _region_filter_particles(pre_df, region_filter, tmp_time, params,
                             df_method, rh_threshold_loader):
    """Select one target region from the prefiltered particles, then apply RH filtering."""
    result_df = region_filter(pre_df)
    return _rh_filter_and_pmass(result_df, tmp_time, params,
                                df_method, rh_threshold_loader)


def _build_tracking_arrays(result, times, df_method, params, blh_lookup):
    """
    Build trajectory arrays from result ([0]=all_P_parts, [1:]=backward-step particle tables).
    blh_lookup(timestamp) -> Series((lat,lon)->BLH_factor) or None (used when df_method='on').

    Returns dict:
      particle (N,), time (T,)
      midpoints lon/lat/z/blh (f32, (T,N)); q (f32, raw); q_diff (f64); f (f64)
      raw lon_raw/lat_raw/z_raw/blh_raw/t/topo (f32, (T,N); t/topo None if absent in input)
      mass/p_mass (f32, (N,)); present (bool, (T,N))
    """
    all_P_parts = result[0]
    particle_indices = all_P_parts.index.values
    N = len(particle_indices)
    n_steps = len(result)
    p_index = pd.Index(particle_indices)

    track_times = pd.DatetimeIndex(
        pd.to_datetime(times, format='%Y%m%d%H') -
        pd.Timedelta(hours=config.TIME_SPAN) * np.arange(n_steps - 1))

    # (n_steps, N) float32 arrays; row 0 from all_P_parts, row j from result[j]
    lon2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    lat2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    z2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    q2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    blh2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    t2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    topo2d = np.full((n_steps, N), np.nan, dtype=np.float32)
    present = np.zeros((n_steps, N), dtype=bool)
    has_t_topo = False

    lon2d[0] = all_P_parts['lon'].values
    lat2d[0] = all_P_parts['lat'].values
    present[0] = True

    for j in range(1, n_steps):
        dfj = result[j]
        if dfj is None or len(dfj) == 0:
            continue
        pos = p_index.get_indexer(dfj.index)
        valid = pos >= 0
        if not valid.all():
            pos = pos[valid]
            dfj = dfj.iloc[valid]
        lon2d[j, pos] = dfj['lon'].values
        lat2d[j, pos] = dfj['lat'].values
        z2d[j, pos] = dfj['z'].values
        q2d[j, pos] = dfj['q'].values
        blh2d[j, pos] = dfj['blh'].values
        if 't' in dfj.columns and 'topo' in dfj.columns:
            t2d[j, pos] = dfj['t'].values
            topo2d[j, pos] = dfj['topo'].values
            has_t_topo = True
        present[j, pos] = True

    # Segment midpoints and q_diff (f32 arithmetic, q_diff stored as f64)
    new_lat = lat2d.copy()
    new_lon = lon2d.copy()
    new_z = z2d.copy()
    new_blh = blh2d.copy()
    new_q_diff = np.zeros((n_steps, N), dtype=np.float64)
    new_q_diff[0] = all_P_parts['q_diff'].values
    if n_steps > 2:
        new_lat[1:-1] = (lat2d[1:-1] + lat2d[2:]) / 2
        new_lon[1:-1] = (lon2d[1:-1] + lon2d[2:]) / 2
        new_z[1:-1] = (z2d[1:-1] + z2d[2:]) / 2
        new_q_diff[1:-1] = q2d[1:-1] - q2d[2:]
        new_blh[1:-1] = (blh2d[1:-1] + blh2d[2:]) / 2

    # Drop the last row (no following step to pair with)
    T = n_steps - 1
    new_lat, new_lon, new_z, new_blh = new_lat[:T], new_lon[:T], new_z[:T], new_blh[:T]
    new_q_diff = new_q_diff[:T]
    q_cut = q2d[:T]

    # f0: per-step evaporation contribution fraction
    if df_method == 'on':
        lat_round = (np.round(new_lat / config.OUTPUT_SPATIAL_RESOLUTION) *
                     config.OUTPUT_SPATIAL_RESOLUTION).astype(np.float64)
        lon_round = (np.round(new_lon / config.OUTPUT_SPATIAL_RESOLUTION) *
                     config.OUTPUT_SPATIAL_RESOLUTION).astype(np.float64)
        blh_factor2d = np.full((T, N), np.nan, dtype=np.float64)
        for trow in range(T):
            lookup = blh_lookup(track_times[trow])
            if lookup is None:
                continue
            keys = pd.MultiIndex.from_arrays([lat_round[trow], lon_round[trow]])
            blh_factor2d[trow] = lookup.reindex(keys).values
        with np.errstate(invalid='ignore', divide='ignore'):
            mask = (new_q_diff > params['q_diff_e']) & (new_z < (blh_factor2d * new_blh))
            f0 = np.where(mask, new_q_diff / q_cut, 0)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            mask = (new_q_diff > params['q_diff_e']) & (new_z < (params['blh_factor'] * new_blh))
            f0 = np.where(mask, new_q_diff / q_cut, 0)

    # f: discount f0 by later uptakes along the trajectory (sequential, order matters)
    f = f0.copy()
    for l in range(2, T):
        f[l:] = f[l:] * (1 - f0[l - 1])

    # Raw values: time[r] = times - r*span corresponds to result[r+1]
    return {
        'particle': particle_indices,
        'time': track_times,
        'lon': new_lon, 'lat': new_lat, 'z': new_z, 'blh': new_blh,
        'q': q_cut, 'q_diff': new_q_diff, 'f': f,
        'lon_raw': lon2d[1:], 'lat_raw': lat2d[1:],
        'z_raw': z2d[1:], 'blh_raw': blh2d[1:],
        't': t2d[1:] if has_t_topo else None,
        'topo': topo2d[1:] if has_t_topo else None,
        'mass': all_P_parts['mass'].values,
        'p_mass': all_P_parts['p_mass'].values,
        'present': present[:T],
    }


def arrays_to_traj_dataset(arr, times):
    """Pack _build_tracking_arrays output into an xarray.Dataset (particle x time)."""
    dims = ('particle', 'time')

    def da(x):
        return xr.DataArray(np.ascontiguousarray(x.T), dims=dims)

    data_vars = {
        # Segment midpoints used by the algorithm
        'lon': da(arr['lon']), 'lat': da(arr['lat']),
        'z': da(arr['z']), 'blh': da(arr['blh']),
        'q': da(arr['q']), 'q_diff': da(arr['q_diff']), 'f': da(arr['f']),
        # Raw instantaneous values
        'lon_raw': da(arr['lon_raw']), 'lat_raw': da(arr['lat_raw']),
        'z_raw': da(arr['z_raw']), 'blh_raw': da(arr['blh_raw']),
        # Per-particle scalars
        'mass': xr.DataArray(arr['mass'], dims=('particle',)),
        'p_mass': xr.DataArray(arr['p_mass'], dims=('particle',)),
    }
    if arr.get('t') is not None:
        data_vars['t'] = da(arr['t'])
        data_vars['topo'] = da(arr['topo'])

    ds = xr.Dataset(
        data_vars,
        coords={'particle': arr['particle'], 'time': arr['time']},
        attrs={
            'description': 'WaterSip backward trajectories of precipitating particles',
            'arrival_time': times,
            'note_midpoint': 'lon/lat/z/blh are segment midpoints used by the WaterSip '
                             'algorithm; row 0 is the arrival (precipitation) segment. '
                             'q is the raw specific humidity; q_diff/f follow WaterSip.',
            'note_raw': 'lon_raw/lat_raw/z_raw/blh_raw/t/topo are instantaneous values '
                        'at each coordinate time.',
            'units': 'lon/lat: degree; z/blh/topo: m; q/q_diff: kg/kg; t: K; '
                     'mass/p_mass: kg; f: 1',
        })
    return ds


def write_traj_nc(ds, times, out_dir=None):
    """Write one trajectory NetCDF (zlib; atomic write)."""
    if out_dir is None:
        out_dir = config.TEMPORARY_FILE_PATH
    out = os.path.join(out_dir,
                       f"{times}-{config.TIME_SPAN}_P_particle_trajectories.nc")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if len(ds['particle']) > 0:
        enc = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
    else:
        enc = None
    tmp = f'{out}.tmp.nc'
    try:
        ds.to_netcdf(tmp, encoding=enc)
    except (ValueError, ImportError, AttributeError, FileNotFoundError):
        # netCDF4 engine unusable (e.g. binary/DLL issues) -> fall back to h5netcdf
        ds.to_netcdf(tmp, encoding=enc, engine='h5netcdf')
    os.replace(tmp, out)
    return out


def make_rh_threshold_loader():
    """Cached loader: tmp_time ('YYYYMMDDHH') -> Series((lat,lon)->RH threshold)."""
    cache = {}

    def loader(tmp_time):
        if tmp_time not in cache:
            rh_file = f'{config.DF_FILE_PATH}/DF_RH_thresholds_{tmp_time}.nc'
            thr = open_nc(rh_file)['DF_RH_thresholds']
            df_thr = thr.to_dataframe(name='RH_threshold').reset_index()
            df_thr = df_thr[(df_thr['RH_threshold'] != 100) & df_thr['RH_threshold'].notna()]
            cache[tmp_time] = pd.Series(
                df_thr['RH_threshold'].values,
                index=pd.MultiIndex.from_arrays(
                    [df_thr['latitude'].values.astype(np.float64),
                     df_thr['longitude'].values.astype(np.float64)]))
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
        return cache[tmp_time]
    return loader


def make_blh_lookup(cache_size=90):
    """Cached loader: Timestamp -> Series((lat,lon)->BLH_factor), None if file missing.
    NaN and zero factors are dropped."""
    cache = {}

    def lookup(t):
        if t not in cache:
            f = f'{config.DF_FILE_PATH}/DF_BLH_factors_{t.strftime("%Y%m%d%H")}.nc'
            if not os.path.exists(f):
                print(f'! Lacking DF_BLH_factors file: {f}')
                cache[t] = None
            else:
                da = open_nc(f)['DF_BLH_factors']
                df0 = da.to_dataframe(name='BLH_factor').reset_index()
                df0 = df0.dropna(subset=['BLH_factor'])
                df0 = df0.query("BLH_factor != 0")
                cache[t] = pd.Series(
                    df0['BLH_factor'].values,
                    index=pd.MultiIndex.from_arrays(
                        [df0['latitude'].values.astype(np.float64),
                         df0['longitude'].values.astype(np.float64)]))
            if len(cache) > cache_size:
                cache.pop(next(iter(cache)))
        return cache[t]
    return lookup


class _Window:
    """Backward-tracking window for one (precipitation time, region); collects per-step particle slices."""
    __slots__ = ('times', 'result', 'remaining', 'time', 'particles', 'out_dir', 'name')

    def __init__(self, times, all_P_parts, n_track_files, out_dir=None, name=None):
        self.times = times                       # 'YYYYMMDDHH'
        self.time = datetime.strptime(times, '%Y%m%d%H')
        self.result = [None] * (n_track_files + 1)
        self.result[0] = all_P_parts
        self.remaining = n_track_files
        self.particles = all_P_parts.index.values
        self.out_dir = out_dir                   # None -> config.TEMPORARY_FILE_PATH
        self.name = name                         # region name (multi-region mode), None otherwise


def _traj_file_path(out_dir, times):
    return os.path.join(out_dir,
                        f"{times}-{config.TIME_SPAN}_P_particle_trajectories.nc")


def _run_filtering_and_tracking(method, df_method, regions, skip_existing=False):
    """
    Particle filtering and moisture tracking for any number of target regions in one
    descending pass over partposit files (read, RH and q_diff prefilter done once).

    regions: [{'name': str or None, 'region': shp path / bounds / None, 'out_dir': str}, ...]
    skip_existing: skip (time, region) pairs whose output exists and shrink the scan range.
    """
    begin_time = time.time()
    region_desc = ', '.join(str(r['name']) for r in regions)
    print(f'\nStep 3 ({method}): Particle filtering and moisture tracking...'
          f' regions: [{region_desc}]')

    params = get_algorithm_params(method)
    span = timedelta(hours=config.TIME_SPAN)
    n_track_files = int(config.TRACKING_DAYS * 24 / config.TIME_SPAN) + 1  # includes the start time

    # ---------- Cached loaders and region filters ----------
    rh_threshold_loader = make_rh_threshold_loader()
    blh_lookup = make_blh_lookup(cache_size=n_track_files + 8)
    region_list = [(r.get('name'), make_region_filter(r['region']), r['out_dir'])
                   for r in regions]

    # ---------- Finalize window: build arrays and write NetCDF ----------
    def finalize_window(w):
        t0 = time.time()
        tag = f' [{w.name}]' if w.name else ''
        try:
            arr = _build_tracking_arrays(w.result, w.times, df_method, params, blh_lookup)
            ds = arrays_to_traj_dataset(arr, w.times)
            write_traj_nc(ds, w.times, out_dir=w.out_dir)
            print(f"Trajectories for {w.times}{tag} established "
                  f"({len(arr['particle'])} particles, {time.time() - t0:.1f}s)")
        except Exception as e:
            # On bad input (e.g. corrupt BLH factor file) write an empty file; downstream skips it
            print(f"! Failed to build trajectories for {w.times}{tag}: {e}")
            write_empty_output(w.times, w.out_dir)

    def write_empty_output(times, out_dir):
        print(f"! Warning: {times} No particles found")
        n_steps_t = n_track_files + 1
        track_times = pd.DatetimeIndex(
            pd.to_datetime(times, format='%Y%m%d%H') -
            pd.Timedelta(hours=config.TIME_SPAN) * np.arange(n_steps_t - 1))
        empty2d = np.zeros((len(track_times), 0), dtype=np.float32)
        arr = {
            'particle': np.array([], dtype=np.int32), 'time': track_times,
            'lon': empty2d, 'lat': empty2d, 'z': empty2d, 'blh': empty2d,
            'q': empty2d, 'q_diff': empty2d.astype(np.float64),
            'f': empty2d.astype(np.float64),
            'lon_raw': empty2d, 'lat_raw': empty2d,
            'z_raw': empty2d, 'blh_raw': empty2d,
            't': empty2d, 'topo': empty2d,
            'mass': np.array([], dtype=np.float32),
            'p_mass': np.array([], dtype=np.float32),
        }
        write_traj_nc(arrays_to_traj_dataset(arr, times), times, out_dir=out_dir)

    # ---------- Plan: regions pending for each precipitation time ----------
    t_first = datetime.strptime(config.START_TIME, '%Y%m%d%H')       # first precipitation time
    t_last_file = datetime.strptime(config.END_TIME, '%Y%m%d%H')     # last file (not a precipitation time)

    reg_map = {}   # datetime -> [(name, filter, out_dir), ...]
    t = t_first
    while t <= t_last_file - span:
        pending = []
        for name, filt, out_dir in region_list:
            if skip_existing and out_dir is not None and \
                    os.path.exists(_traj_file_path(out_dir, t.strftime('%Y%m%d%H'))):
                continue
            pending.append((name, filt, out_dir))
        if pending:
            reg_map[t] = pending
        t += span

    if not reg_map:
        print('Step 3: all trajectory files already exist, nothing to do.')
        return

    # Limit scan range to pending times (avoids re-reading files when resuming)
    t_scan_start = max(reg_map) + span                                # latest file needed
    t_scan_end = min(reg_map) - timedelta(days=config.TRACKING_DAYS)  # earliest backward file needed
    n_files_total = int((t_scan_start - t_scan_end) / span) + 1

    # ---------- Main loop: descending time, each file read once ----------
    active = {}      # (datetime, name) -> _Window
    prev_raw = None  # raw particle table (with RH) from the previous, one-TIME_SPAN-later file
    n_done = 0

    t_cur = t_scan_start
    while t_cur >= t_scan_end:
        times_cur = t_cur.strftime('%Y%m%d%H')
        file_path = os.path.join(config.PARTPOSIT_PATH, f"partposit_{t_cur.strftime('%Y%m%d%H')}0000")

        try:
            fdf_raw = readpartposit_to_df(
                file_path, variables=['lon', 'lat', 'z', 'q', 'blh', 'mass', 't', 'dens', 'topo'])
            calculate_RH(fdf_raw)
        except Exception as e:
            print(f"\n! Process file {file_path} Failure: {e}")
            fdf_raw = None

        # 1) Pending precipitation time: prefilter once, then open a window per region
        if t_cur in reg_map:
            if prev_raw is None or fdf_raw is None:
                print(f'! get_and_track_particles generated an exception: '
                      f'missing partposit data for {times_cur}')
            else:
                pre_df = None
                try:
                    pre_df = _prefilter_particles(prev_raw, fdf_raw, times_cur, params)
                except Exception as e:
                    print(f'! get_and_track_particles generated an exception: {e}')
                if pre_df is not None:
                    for name, filt, out_dir in reg_map[t_cur]:
                        tag = f' [{name}]' if name else ''
                        try:
                            all_P_parts = _region_filter_particles(
                                pre_df, filt, times_cur, params, df_method, rh_threshold_loader)
                            print(f"\nStart track particles for {times_cur}{tag}, "
                                  f"particle number: {len(all_P_parts)}")
                            if len(all_P_parts) == 0:
                                write_empty_output(times_cur, out_dir)
                            else:
                                active[(t_cur, name)] = _Window(
                                    times_cur, all_P_parts, n_track_files, out_dir, name)
                        except Exception as e:
                            print(f'! get_and_track_particles generated an '
                                  f'exception{tag}: {e}')
                del pre_df

        # 2) Extract particle slices for every active window covering this file
        if fdf_raw is not None:
            fdf_slice = fdf_raw[_SLICE_COLS]
            fidx = fdf_raw.index
            fidx_unique = fidx.is_unique
        for w_key in list(active.keys()):
            w = active[w_key]
            idx = int((w.time - t_cur) / span)      # backward step index
            if idx < 0 or idx >= n_track_files:
                continue
            if fdf_raw is None:
                sl = pd.DataFrame(columns=_SLICE_COLS)
            else:
                if fidx_unique:
                    pos = fidx.get_indexer(w.particles)
                    pos = np.sort(pos[pos >= 0])    # sorted positions keep file row order
                    has_rows = len(pos) > 0
                else:  # duplicate particle IDs: fall back to a boolean mask
                    mask = fidx.isin(w.particles)
                    pos = np.flatnonzero(mask)
                    has_rows = pos.size > 0
                if has_rows:
                    sl = fdf_slice.take(pos)
                    sl.index.name = 'index'
                else:
                    sl = pd.DataFrame(columns=_SLICE_COLS)
            w.result[idx + 1] = sl
            w.remaining -= 1
            if w.remaining == 0:
                finalize_window(w)
                del active[w_key]
                n_done += 1

        prev_raw = fdf_raw
        n_read = int((t_scan_start - t_cur) / span) + 1
        print(f"[{n_read}/{n_files_total}] {times_cur} processed, "
              f"active windows: {len(active)}, done: {n_done}")
        t_cur -= span
        gc.collect()

    # Finalize any incomplete windows left after the scan (should not normally happen)
    for w_key in sorted(active.keys(), key=lambda k: k[0], reverse=True):
        w = active[w_key]
        print(f'! Warning: window {w.times} incomplete ({w.remaining} files missing)')
        for k in range(1, len(w.result)):
            if w.result[k] is None:
                w.result[k] = pd.DataFrame(columns=_SLICE_COLS)
        finalize_window(w)
        n_done += 1

    print(f"Step 3 done, total {time.time() - begin_time:.1f} seconds")


def filtering_and_tracking(method, df_method):
    """Single-region run using config.TARGET_REGION and config.TEMPORARY_FILE_PATH."""
    regions = [{'name': None, 'region': config.TARGET_REGION,
                'out_dir': config.TEMPORARY_FILE_PATH}]
    _run_filtering_and_tracking(method, df_method, regions, skip_existing=False)


if __name__ == "__main__":
    filtering_and_tracking('WaterSip-DF-TTMC', 'on')  # 'WaterSip-DF-TTMC', 'on' ; 'WaterSip', 'off'
