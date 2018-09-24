#!/usr/bin/env python

""" A collection of scripts for processing GPS streams"""

import ast
from collections import *
import datetime as dt
import multiprocessing as mul
import os
from pathlib import Path
import re

import googlemaps
from googlemaps.exceptions import *
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import DBSCAN


__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'

"""Google Maps place types to ignore 
https://developers.google.com/places/supported_types
"""
IGNORED_PLACE_TYPES = [
    'administrative_area_level',
    'administrative_area_level_1',
    'administrative_area_level_2',
    'administrative_area_level_3',
    'administrative_area_level_4',
    'administrative_area_level_5',
    'country',
    'route',
    'street_address',
    'street_number',
    'sublocality',
    'sublocality_level_5',
    'sublocality_level_4',
    'sublocality_level_3',
    'sublocality_level_2',
    'sublocality_level_1',
    'subpremise',
    'locality',
    'political'
]

"""Google Maps place fields to request
 https://developers.google.com/places/web-service/details
 """
PLACE_QUERY_FIELDS = ['name', 'type', 'rating', 'price_level', 'geometry']

"""ensure the user has an API key to Google Maps"""
try:
    PLACES_KEY = os.environ['GMAPS_PLACES_KEY']
    print(f'Google Places API Key: {PLACES_KEY}')
except KeyError:
    try:
        import brightenv2.brighten_secrets as bs
        PLACES_KEY = bs.PLACES_KEY
    except ModuleNotFoundError:
        PLACES_KEY = None
        print('WARNING: No API key found to access Google Maps.')

"""named tuple used to clarify types used in the scripts below"""
GPS = namedtuple('GPS', ['lat', 'lon', 'ts'])
Cluster = namedtuple('Cluster', ['lat', 'lon', 'cid'])


def as_pydate(date):
    ismil = True

    if any(s in date for s in ['a.m.', 'AM', 'a. m.']):
        date = re.sub(r'(a.m.|a. m.|AM)', '', date).strip()

    if any(s in date for s in ['p.m.', 'p. m.', 'PM']):
        date = re.sub(r'(p.m.|p. m.|PM)', '', date).strip()
        ismil = False

    if '/' in date:
        date = re.sub('/', '-', date)

    try:
        date = dt.datetime.strptime(date, '%d-%m-%y %H:%M')

    except ValueError as e:
        argstr = ''.join(e.args)

        if ':' in argstr:
            date = dt.datetime.strptime(date, '%d-%m-%y %H:%M:%S')
        else:
            raise e

    date = date + dt.timedelta(hours=12) if not ismil else date
    return date


# --------------------------------------------------------------------------
# GPS and clustering
# --------------------------------------------------------------------------
def cluster_metrics(clusters, entries):
    if 'cnt' in clusters.columns:
        clusters.drop(columns='cnt', inplace=True)
    else:
        pass

    stats, dfs = [], []

    grouped = entries.groupby('cid')
    for name, group in grouped:
        df = pd.DataFrame(group)

        df['day'] = df.time_in.apply(
            lambda r: dt.date(year=r.year, month=r.month, day=r.day)
        )
        df['weekday'] = df.day.apply(lambda r: r.weekday())
        df['week']   = df.day.apply(
            lambda r: f'{r.year}{r.isocalendar()[1]}'
        )
        df['month']  = df.day.apply(lambda r: r.month)
        df['ymonth'] = df.day.apply(lambda r: f'{r.year}{r.month}')
        df['in_tod'] = df.time_in.apply(
            lambda r: dt.time(hour=r.hour, minute=r.minute)
        )
        df['in_hod'] = df.time_in.apply(lambda r: r.hour)
        df['out_hod'] = df.time_out.apply(lambda r: r.hour)
        df['out_tod'] = df.time_out.apply(
            lambda r: dt.time(hour=r.hour, minute=r.minute)
        )

        groups_by_itod = df.groupby('in_hod')
        groups_by_itod = [(n, pd.DataFrame(g)) for n, g in groups_by_itod]
        counts_by_itod = sorted(
            [(g[0], len(g[1])) for g in groups_by_itod],
            key=lambda t: t[1],
            reverse=True
        )[:3]

        groups_by_otod = df.groupby('out_hod')
        groups_by_otod = [(n, pd.DataFrame(g)) for n, g in groups_by_otod]
        counts_by_otod = sorted(
            [(g[0], len(g[1])) for g in groups_by_otod],
            key=lambda t: t[1],
            reverse=True
        )[:3]

        groups_by_day = df.groupby('day')
        groups_by_day = [pd.DataFrame(g) for n, g in groups_by_day]
        daily_counts  = [len(g) for g in groups_by_day]

        groups_by_weekday = df.groupby('weekday')
        groups_by_weekday = [(n, pd.DataFrame(g)) for n, g in groups_by_weekday]
        counts_by_dow = sorted(
            [(g[0], len(g[1])) for g in groups_by_weekday],
            key=lambda t: t[1],
            reverse=True
        )[:3]

        groups_by_week = df.groupby('week')
        groups_by_week = [pd.DataFrame(g) for n, g in groups_by_week]
        weekly_counts  = [len(g) for g in groups_by_week]

        groups_by_month = df.groupby('month')
        groups_by_month = [(n, pd.DataFrame(g)) for n, g in groups_by_month]
        counts_by_month = sorted(
            [(g[0], len(g[1])) for g in groups_by_month],
            key=lambda t: t[1],
            reverse=True
        )[:3]

        groups_by_ymonth = df.groupby('ymonth')
        groups_by_ymonth = [pd.DataFrame(g) for n, g in groups_by_ymonth]
        ymonthly_counts  = [len(g) for g in groups_by_ymonth]

        stats.append(
            dict(
                cid=name,
                times_entered=len(df),
                total_duration=np.round(df.duration.sum().total_seconds()/3600, 3),
                mean_duration=np.round(df.duration.mean().total_seconds()/3600, 3),
                std_duration=np.round(df.duration.std().total_seconds()/3600, 3),
                max_duration=np.round(df.duration.max().total_seconds()/3600, 3),
                min_duration=np.round(df.duration.min().total_seconds()/3600, 3),
                earliest_time_in=df.in_tod.min(),
                latest_time_in=df.in_tod.max(),
                earliest_time_out=df.out_tod.min(),
                latest_time_out=df.out_tod.max(),
                days_entered=int(len(daily_counts)),
                count_before_9=int(np.sum([
                    1 for r in df.in_tod if r < dt.time(hour=9)
                ])),
                count_between_9_12=int(np.sum([
                    1 for r in df.in_tod
                    if dt.time(hour=9) <= r < dt.time(hour=12)
                ])),
                count_between_12_5=int(np.sum([
                    1 for r in df.in_tod
                    if dt.time(hour=5) <= r < dt.time(hour=17)
                ])),
                count_after_5=int(np.sum([
                    1 for r in df.in_tod if dt.time(hour=17) <= r
                ])),
                max_entries_per_day=int(np.max(daily_counts)),
                min_entries_per_day=int(np.min(daily_counts)),
                mean_entries_per_day=np.round(np.mean(daily_counts), 3),
                std_entries_per_day=np.round(np.std(daily_counts), 3),
                max_entries_per_week=int(np.max(weekly_counts)),
                min_entries_per_week=int(np.min(weekly_counts)),
                mean_entries_per_week=np.round(np.mean(weekly_counts), 3),
                std_entries_per_week=np.round(np.std(weekly_counts), 3),
                max_entries_per_month=int(np.max(ymonthly_counts)),
                min_entries_per_month=int(np.min(ymonthly_counts)),
                mean_entries_per_month=np.round(np.mean(ymonthly_counts), 3),
                std_entries_per_month=np.round(np.std(ymonthly_counts), 3),
                most_common_hod_in=int(counts_by_itod[0][0]),
                most_common_hod_out=int(counts_by_otod[0][0]),
                most_common_dow=int(counts_by_dow[0][0]),
                most_common_dow_cnt=int(counts_by_dow[0][1]),
                most_common_month=int(counts_by_month[0][0]),
                most_common_month_cnt=int(counts_by_month[0][1]),
            )
        )


    if len(stats) > 0:
        stats = pd.DataFrame().from_dict(stats)
        stats.set_index('cid', inplace=True)

        clusters = clusters.join(stats, on='cid', how='outer', sort=True)
        return clusters
    else:
        return None


def process_velocities(gps_records):
    cols = ['lat', 'lon', 'ts']
    records = pd.DataFrame(gps_records, columns=cols)

    # generate a 'nan' row for the first coordinate
    nanrow = [dict(
        displacement=np.nan,
        time_delta=np.nan,
        velocity=np.nan,
        binning='stationary')
    ]

    # rolling window calculating between rows i and i-1
    x = list(map(
        lambda t: discrete_velocity(*t),
        [(
            tuple((v for v in records.loc[i, cols].values)),
            tuple((v for v in records.loc[i-1, cols].values))
        )
            for i in range(1, len(records))
        ]))

    # merge in the new columns
    x = pd.DataFrame(nanrow + x)
    records = pd.concat([records, x], axis=1)
    return records


def discrete_velocity(coordinate_a, coordinate_b):
    """converts the velocity between two points into a discrete value

    Notes:
        For a more detailed explanation see notebooks/determining_transportation_speeds.ipynb
        paper: https://peerj.com/articles/4640/
        data: https://figshare.com/articles/A_public_data_set_of_overground_and_treadmill_walking_kinematics_and_kinetics_of_healthy_individuals/5722711

    Args:
        coordinate_a: (float, float, dt.datetime) lat, lon, timestamp
        coordinate_b: (float, float, dt.datetime) lat, lon, timestamp

    Returns:
        dictionary: {
        'displacement': float,
        'time_delta': int,
        'mps': float,
        'binning': str
        }

    Raises:
        ValueError if supplied tuples are incorrect

    Notes:
        available bins: { 'stationary', 'walking', 'brunch', 'powered_vehicle', 'high_speed_transportation', 'anomaly' }
    """
    if not (isinstance(coordinate_a[-1], dt.datetime) and isinstance(coordinate_b[-1], dt.datetime)):
        raise TypeError('the third argument of each tuple must be a datetime')

    meters = geo_distance(*coordinate_a[:2], *coordinate_b[:2])

    if coordinate_b[-1] > coordinate_a[-1]:
        seconds = (coordinate_b[-1] - coordinate_a[-1]).seconds
    else:
        seconds = (coordinate_a[-1] - coordinate_b[-1]).seconds

    velocity = meters/seconds if seconds != 0 else np.nan

    # stationary - if the velocity is less than what you'd expect
    # the standard https://www.gps.gov/systems/gps/performance/accuracy/
    if 0 <= meters < 4.9:
        binning = 'stationary'

    # determined by the distribution as shown in the Jupyter notebook shown
    # above.
    elif 0 < velocity < .7:
        binning = 'stationary'

    elif velocity < 1.5:
        binning = 'walking'

    # right now I'm assuming anything over 20mph is most likely a powered
    # vehicle. I'm having difficulty finding a
    # good data set for this. 20 mph ~ 8.9 m/s
    elif velocity < 6.9:
        binning = 'brunch'

    # 67.056 m/s ~ 150 mph ~ 241.4016 kmh
    elif velocity < 67.056:
        binning = 'powered_vehicle'

    # anything else reasonable must be some sort of high speed transit
    # 312.928 m/s ~ 700 mph ~ 1126.54 kph
    elif velocity < 312.928:
        binning = 'high_speed_transportation'

    else:
        binning = 'anomaly'

    return dict(
        displacement=np.round(meters, 1),
        time_delta=seconds,
        velocity=np.round(velocity, 3),
        binning=binning
    )


def estimate_home_location(records, parameters=None):
    """calculate approximate home location from GPS records

    Args:
        records: (DataFrame)
        parameters: (dict)

    Returns:
        { lat: float, lon: float }
    """
    gpr = [
        (i, GPS(g.lat, g.lon, g.ts)) for i, g in enumerate(records.itertuples())
        if (17 < g.ts.hour or g.ts.hour < 9)
    ]

    args = gps_dbscan([t[1] for t in gpr], parameters=parameters)
    home, labels = __top_cluster(args)

    if home is not None:
        t = []
        for i, label in enumerate(labels):
            if label == home.get('cid'):
                t.append(gpr[i][0])

        return home, t
    else:
        return None, []


def estimate_work_location(records, parameters=None):
    """calculate approximate work location from GPS records

    Args:
        records: (DataFrame)
        parameters

    Returns:
        Cluster, [int]
    """
    gpr = [
        (i, GPS(g.lat, g.lon, g.ts)) for i, g in
        enumerate(records.itertuples())
        if not (17 <= g.ts.hour or g.ts.hour < 9)
           and g.ts.weekday() < 5
    ]

    args = gps_dbscan([t[1] for t in gpr], parameters=parameters)
    work, labels = __top_cluster(args)

    if work is not None:
        t = []
        for i, label in enumerate(labels):
            if label == work.get('cid'):
                t.append(gpr[i][0])

        return work, t
    else:
        return None, []


def extract_cluster_centers(gps_records, dbscan):
    # extract cluster labels, bincount, and select the top cluster
    clusters = np.unique(dbscan.labels_)

    # process each cluster center
    centers = []
    for ci in clusters:
        if ci == -1:
            continue

        # find the indices for this center
        idx = [i for i, k in enumerate(dbscan.labels_) if k == ci]

        # extract the gps points in this cluster
        records = pd.DataFrame(
            gps_records,
            index=np.arange(len(gps_records)),
            columns=['lat', 'lon', 'ts']
        ).iloc[idx, :]

        # lat stats
        latmean = records.lat.mean()
        latmax = records.lat.max()
        latmin = records.lat.min()
        latrange = latmax-latmin
        latIQR = np.percentile(records.lat.values, [.25, .75])
        latIQR = latIQR[1] - latIQR[0]
        latSTD = records.lat.std()

        # lon stats
        lonmean = records.lon.mean()
        lonmax = records.lon.max()
        lonmin = records.lon.min()
        lonrange = lonmax-lonmin
        lonIQR = np.percentile(records.lon.values, [.25, .75])
        lonIQR = lonIQR[1] - lonIQR[0]
        lonSTD = records.lon.std()

        max_from_center = np.max([
            geo_distance(r.lat, r.lon, latmean, lonmean)
            for r in records.itertuples()
        ])

        # calculate and append the centeroid
        centers.append(dict(
            lat=np.round(latmean, 5),
            lon=np.round(lonmean, 5),
            cid=ci,
            lat_range=np.round(latrange, 5),
            lat_IQR=np.round(latIQR, 5),
            lat_min=np.round(latmin, 5),
            lat_max=np.round(latmax, 5),
            lat_std=np.round(latSTD, 5),
            lon_range=np.round(lonrange, 5),
            lon_IQR=np.round(lonIQR, 5),
            lon_min=np.round(lonmin, 5),
            lon_max=np.round(lonmax, 5),
            lon_std=np.round(lonSTD, 5),
            max_distance_from_center=np.round(max_from_center, 3)
        ))

    return centers


def geo_distance(lat1, lon1, lat2, lon2):
    """calculates the geographic distance between coordinates
    https://www.movable-type.co.uk/scripts/latlong.html

    Args:
        lat1: (float)
        lon1: (float)
        lat2: (float)
        lon2: (float)
        metric: (str) in { 'meters', 'km', 'mile' }

    Returns:
        float representing the distance in meters
    """
    r = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return r*c*1000


def __wrapit(args):
    return geo_distance(*args)


def geo_pairwise_distances(x, n_jobs=1):
    """

    Args:
        x: [(float, float),]
        n_jobs: number of cores to use, -1 for all

    Returns:
        [float,]
    """
    x = pd.DataFrame(x)
    x['key'] = 0
    x = pd.merge(x, x, on='key', how='outer').drop(columns='key')

    x = [(xi[1], xi[2], xi[3], xi[4]) for xi in x.itertuples()]

    cpus = mul.cpu_count()
    cpus = cpus if n_jobs <= 0 or n_jobs >= cpus else n_jobs
    pool = mul.Pool(cpus)

    result = sorted(list(pool.map(__wrapit, x)))

    pool.close()
    pool.join()
    return np.array(result)


def get_clusters_with_context(records, parameters=None):
    records['cid'] = 'xNot'
    stationary = records.loc[records.binning == 'stationary']
    others = records.loc[records.binning != 'stationary']

    if len(records) < 3:
        return records, None

    # find the home records
    home, hmask = estimate_home_location(stationary, parameters)
    home_records = stationary.iloc[hmask, :].copy()
    home_records.cid = 'home'

    # find the work records
    work, wmask = estimate_work_location(stationary, parameters)
    work_records = stationary.iloc[wmask, :].copy()
    work_records.cid = 'work'

    # scan the remaining records
    rmask = list(set(np.arange(stationary.shape[0])) - set(hmask+wmask))
    remaining = stationary.iloc[rmask, :].copy()
    gps_records = [
        GPS(lat=r.lat, lon=r.lon, ts=r.ts) for r in remaining.itertuples()
    ]
    labels, clusters = gps_dbscan(gps_records, parameters)

    remaining.cid = [
        f'x{l}' if l != -1 else 'xNot' for l in labels
    ]

    # append the home and work clusters
    if home is not None:
        home['cid'] = 'home'
        clusters += [home]

    if work is not None:
        work['cid'] = 'work'
        clusters += [work]

    clusters = pd.DataFrame(clusters)

    clusters.cid = [
        f'x{l}' if l not in ['home', 'work'] else l
        for l in clusters.cid
    ] if 'cid' in clusters.columns else 'xNot'

    records = pd.concat([home_records, work_records, remaining, others])

    names, types = [], []
    for ci in clusters.itertuples():
        r = request_nearby_places(ci.lat, ci.lon)
        content = r.get('content') if r is not None else None

        if content is not None and content.get('status') == 'OK':
            results = content.get('results')

            n = results[0].get('name')
            t = ', '.join(results[0].get('types'))
            names.append(n)
            types.append(t)

        else:
            names.append('nap')
            types.append('nap')

    clusters['name'] = names
    clusters['categories'] = types
    records['distance_from_home'] = [
        np.round(geo_distance(
            home.get('lat'),
            home.get('lon'),
            r.lat, r.lon), 3)
        if home is not None else np.nan
        for r in records.itertuples()
    ]

    assert len(records.loc[records.cid == '', :]) == 0
    return records, clusters


def get_cluster_times(records, clusters):
    added_yday = False

    if 'yday' not in records.columns:
        records['yday'] = records.ts.apply(
            lambda t: f'{t.year}{t.timetuple().tm_yday}'
        )
        added_yday = True
    else:
        pass

    grouped = records.groupby(['yday'])

    ln_c, ln_d, entries = None, 0, []
    for yday, group in grouped:
        # convert to dataframe and sort by timestamp
        df = pd.DataFrame(group).sort_values('ts', ascending=True)

        start = end = last_c, ts_cnt = None, 0
        for i, r in enumerate(df.itertuples()):

            # if this point is assigned to a cluster
            if r.cid != 'xNot':
                c = list(clusters.loc[clusters.cid == r.cid].itertuples())[0]
                c = Cluster(c.lat, c.lon, c.cid)

                # if a start hasn't been recorded
                if start is None:

                    # either this is a continuation of the day prior
                    if i == 0 \
                            and ln_c is not None \
                            and yday == ln_d + 1 \
                            and c.cid == ln_c.cid:

                        # make sure this timestamp starts at midnight
                        start = end = GPS(
                            lat=c.lat,
                            lon=c.lon,
                            ts=dt.datetime(
                                year=r.ts.year,
                                month=r.ts.month,
                                day=r.ts.day)
                        )

                        # and last night's timestamp goes to midnight
                        to = entries[-1].time_out
                        entries[-1].time_out = dt.datetime(
                            year=to.year, month=to.month, day=to.day,
                            hour=23, minute=59, second=59
                        )

                        last_c = c
                        ts_cnt += 1

                    # or the subject started recording after they left
                    # wherever they went to sleep or more than a day has
                    # passed since the last recording
                    else:
                        start = end = r
                        last_c = c
                        ts_cnt += 1

                # if we're still in the same cluster reset the last position
                elif last_c is not None and last_c.cid == c.cid:
                    end = r
                    ts_cnt += 1

                # otherwise they've gone to a different cluster
                else:
                    # make sure they weren't just passing through
                    if ts_cnt > 1:
                        entries.append(
                            dict(
                                cid=last_c.cid,
                                time_in=start.ts,
                                time_out=end.ts,
                                lat=last_c.lat,
                                lon=last_c.lon
                            )
                        )
                    else:
                        pass

                    start = end = r
                    last_c = c
                    ts_cnt = 1

            # the subject left a valid cluster
            elif start is not None:
                if ts_cnt > 1:
                    entries.append(
                        dict(
                            cid=last_c.cid,
                            time_in=start.ts,
                            time_out=end.ts,
                            lat=last_c.lat,
                            lon=last_c.lon
                        )
                    )
                else:
                    pass

                start = end = last_c = None
                ts_cnt = 0

            # and finally, if the day started with a point that hasn't been
            # assigned to cluster
            else:
                last_c = None
                continue

        # reset last-night's cluster and day
        ln_c, ln_d = last_c, int(yday)

    if added_yday:
        records.drop(columns=['yday'], inplace=True)
    else:
        pass

    entries = pd.DataFrame(entries)
    entries['duration'] = [
        r.time_out-r.time_in for r in entries.itertuples()
    ]
    entries['midpoint'] = [
        r.time_in + r.duration/2
        for r in entries.itertuples()
    ]
    entries['date'] = entries.midpoint.apply(lambda r: r.date())
    entries['tod'] = entries.midpoint.apply(lambda r: r.time())

    def tod_bin(x):
        x = x.hour

        if x < 9:
            return 'early_morning'
        elif 9 <= x < 12:
            return 'morning'
        elif 12 <= x < 17:
            return 'afternoon'
        elif 17 <= x:
            return 'evening'

    entries['tod_bin'] = entries.midpoint.apply(tod_bin)

    def cname(cid):
        c = clusters.loc[clusters.cid == cid, 'name'].values

        if len(c) > 0:
            return c[0]

        return 'err'

    if 'cid' not in entries.columns:
        return None
    else:
        entries['cname'] = entries.cid.apply(cname)

    return entries


def gps_dbscan(gps_records, parameters=None, optics=False):
    """perform DBSCAN and return cluster with most number of components
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Args:
        gps_records: [GPS]
        parameters: (dict) optional parameters for DBSCAN. See above
        optics: (bool)

    Returns:
        labels, [dict(lat, lon, count, records)]
        ordered by number of points contained in each cluster
    """
    if len(gps_records) < 2:
        return [], []

    # check if parameters were supplied
    eps, min_samples, metric, n_jobs = __validate_scikit_params(parameters)

    # perform the clustering
    if optics:
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs
        ).fit([(g.lat, g.lon) for g in gps_records])
    else:
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs
        ).fit([(g.lat, g.lon) for g in gps_records])

    clusters = extract_cluster_centers(gps_records, dbscan)
    return dbscan.labels_, clusters


def impute_between(coordinate_a, coordinate_b, freq):
    """

    Args:
        coordinate_a:
        coordinate_b:
        freq:

    Returns:

    """
    metrics = discrete_velocity(coordinate_a, coordinate_b)
    if metrics.get('binning') != 'stationary' and \
            metrics.get('displacement') > 100:
        return

    a_lat, a_lon, a_ts = coordinate_a
    b_lat, b_lon, b_ts = coordinate_b

    if not (isinstance(a_ts, dt.datetime) and isinstance(b_ts, dt.datetime)):
        raise TypeError('third element of each coordinate tuple must be dt')

    if (b_ts-a_ts).seconds < 15*60:
        return

    fill_range = list(pd.date_range(a_ts, b_ts, freq=freq))

    # ensure the returned dataframe range is exclusive
    if fill_range[0] == a_ts:
        fill_range.remove(fill_range[0])
    if fill_range[-1] == b_ts:
        fill_range.remove(fill_range[-1])

    fill_lat = np.linspace(a_lat, b_lat, len(fill_range))
    fill_lon = np.linspace(a_lon, b_lon, len(fill_range))

    t = dict(lat=fill_lat, lon=fill_lon, ts=fill_range)
    return pd.DataFrame(t)


def impute_stationary_coordinates(records, freq='10Min', metrics=True):
    """resample a stream of `gps_records` boosting the number of points
    spent at stationary positions. This method is used due to to how the
    data were collected. GPS coordinates were recorded at either 15min
    intervals or when the user moved 100 meters, whichever came first.
    For this reason, the data is collected in a biased manner. Using this
    function, we even the distribution across semantically binned velocities
    making density based clustering algorithms more effective.

    Args:
        records: (DataFrame)
        freq: (str)
        metrics: (bool)

    Returns:
        [GPS]
    """
    records.sort_values('ts', inplace=True)

    # make a column unique by day to group on
    records['day'] = records.ts.apply(lambda r: r.date())

    # group by days
    grouped, all_days = records.groupby(['day']), []
    for name, group in grouped:
        records = pd.DataFrame(group)
        records.index = np.arange(len(records))

        if len(records) == 1:
            continue

        # fill between the stationary gaps
        cols = ['lat', 'lon', 'ts']
        x = list(map(
            lambda t: impute_between(*t, freq),
            [(
                tuple((v for v in records.loc[i-1, cols].values)),
                tuple((v for v in records.loc[i, cols].values))
            )
                for i in range(1, len(records))
            ]
        ))

        # flatten the list of dataframes and merge with the main
        x = [xi for xi in x if xi is not None]

        if len(x) > 0:
            x = pd.concat(x)
            records = pd.concat([records, x], axis=0)
            records.sort_values('ts', inplace=True)

            # recalculate velocities between the imputed locations
            records = process_velocities(
                [(r.lat, r.lon, r.ts) for r in records.itertuples()]
            )
        else:
            pass

        all_days.append(records)

    # combine, sort, reset index
    if len(all_days) > 0:
        records = pd.concat(all_days, axis=0)
        records.sort_values('ts', inplace=True)
        records.index = np.arange(len(records))

    # cleanup
    if 'day' in records.columns:
        records.drop(columns='day', inplace=True)
    else:
        pass

    if not metrics:
        records.drop(
            columns=['binning', 'displacement', 'time_delta', 'velocity']
        )
    else:
        pass

    return records


def resample_gps_intervals(records):
    if any([
        True if c not in ['lat', 'lon', 'ts'] else False
        for c in records.columns
    ]):
        raise Exception('frame should only include lat, lon, ts')

    def gx(row):
        gv = row.gv

        return dt.datetime(
            year=gv.year, month=gv.month, day=gv.day,
            hour=gv.hour, minute=gv.minute
        )

    records['gv'] = records.ts.apply(lambda r: dt.datetime(
        year=r.year, month=r.month, day=r.day, hour=r.hour, minute=r.minute
    ))
    resampled = records.groupby(['gv']).mean().reset_index()

    resampled['ts'] = list(map(gx, resampled.itertuples()))

    resampled.drop(columns='gv', inplace=True)
    return resampled


# --------------------------------------------------------------------------
# Google Places requests
# --------------------------------------------------------------------------
def request_nearby_places(lat, lon, radius=50):
    return
    """retrieve an ordered list of nearby by places

    Notes:
        results are returned in Google prominence order

    Args:
        lat: (float) latitude in decimal-degree (DD) format
        lon: (float) longitude in DD format
        radius: (float) radius in meters for which to retrieve places

    Raises:
        ValueError: if -90 > lat > 90 OR -180 > lon > 180
        EnvironmentError: if no Google Maps API Key is found

    Returns:
        [brightenv2.context.Place]
    """
    __verify_key()

    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise ValueError('lat, lon must be in a valid range')

    # get the cache
    cache_fn = 'gmaps_nearby_cache.csv'
    cache_fn = os.path.join(Path.home(), '.brighten', cache_fn)
    try:
        cache = pd.read_csv(cache_fn)
    except FileNotFoundError:
        cache = pd.DataFrame(columns=['lat', 'lon', 'content'])

    lat = np.round(lat, 5)
    lon = np.round(lon, 5)

    # first check our cache to see of we've made the same request
    nearby = cache.loc[
        (np.isclose(cache.lat, lat)) & (np.isclose(cache.lon, lon))
    ] if len(cache) > 0 else []

    if len(nearby) > 0:
        return dict(
            content=ast.literal_eval(nearby.content.values[0]),
            cache='hit'
        )

    else:
        # get nearby places from Google
        gmaps = googlemaps.Client(key=PLACES_KEY)
        nearby_request = gmaps.places_nearby(
            location=(lat, lon),
            radius=radius,
            rank_by='prominence'
        )

        result = dict(lat=lat, lon=lon, content=str(nearby_request))

        # make sure the cache has the results
        cache.loc[-1] = result
        cache.to_csv(cache_fn, index=None)

        return dict(content=nearby_request, cache='miss')


def request_place_details(place_id):
    return
    """make a detailed place request through Google Maps

    Args:
        place_id: (str) representing the Google place_id

    Raises:
        EnvironmentError: if no Google Maps API Key is found

    Returns:
        brightenv2.context.Place or None
    """
    __verify_key()

    client = googlemaps.Client(key=PLACES_KEY)

    # make the API call
    try:
        place_request = client.place(place_id, fields=PLACE_QUERY_FIELDS)
    except ApiError as e:
        print(f'The request failed with response: '
              f'{", ".join([str(a) for a in e.args if a is not None])}')
        return None

    # parse the place details into brightenv2.context.Place
    place = None
    if place_request.get('status') == 'OK':
        result = place_request.get('result')
        location = result.get('geometry').get('location')

        categories = result.get('types')
        name = result.get('name')
        lat = location.get('lat')
        lon = location.get('lng')
        rating = result.get('rating')

        if not STANDALONE:
            categories = [
                ctx.Category(name=name) for name in result.get('types')
            ]

            place = ctx.Place(
                place_id=place_id,
                name=name,
                lat=lat,
                lon=lon,
                rating=rating,
                categories=categories,
                last_sync=dt.datetime.now()
            )
        else:
            place = dict(
                place_id=place_id,
                name=name,
                lat=lat,
                lo=lon,
                rating=rating,
                categories=categories
            )
    else:
        pass

    return place


# --------------------------------------------------------------------------
# Private Methods
# --------------------------------------------------------------------------
def __top_cluster(args):
    if args is not None and len(args) > 0:
        try:
            labels, clusters = args
            lmax = mode([i for i in labels if i != -1]).mode[0]

            c = [c for c in clusters if c.get('cid') == lmax]
            c = c[0] if len(c) > 0 else None
            return c, labels

        except IndexError:
            return None, args[0]
    else:
        return None, args[0]


def __validate_scikit_params(parameters):
    """validate scikit-learn DBSCAN parameters

    Args:
        parameters: (dict)

    Returns:
        eps, min_samples, metric, n_jobs
    """
    eps, min_samples, metric, n_jobs = None, None, None, None

    if parameters is not None:
        eps = parameters.get('eps')
        min_samples = parameters.get('min_samples')
        metric = parameters.get('metric')
        n_jobs = parameters.get('n_jobs')
    else:
        pass

    # set default values if the dictionary didn't contain correct keys
    eps = 0.001 if eps is None else eps
    min_samples = 2 if min_samples is None else min_samples
    metric = 'euclidean' if metric is None else metric
    n_jobs = -1 if n_jobs is None else n_jobs

    return eps, min_samples, metric, n_jobs


def __verify_key():
    if PLACES_KEY is None:
        raise EnvironmentError(
            'Environment variable: GMAPS_PLACES_KEY not found. Cannot '
            'complete request.'
        )
    else:
        pass


if __name__ == '__main__':
    pass
