#!/usr/bin/env python

""" A collection of common scripts imported by both `gscap.gps' and
`gscap.weather`. They are used extensively throughout both of those modules
but have plenty of use cases for every day use!
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'


"""check for config file"""
config_file = os.path.join(Path.home(), '.gscapConfig')
if not os.path.exists(config_file):
    print('configuration file not found')
else:
    with open(config_file, 'r') as f:
        cf = f.readlines()

    # read each line of the file into a dictionary as a key value pair separated with an '='
    #  ignore lines beginning with '#'
    CONFIG = {k: v for k, v in [list(map(lambda x: x.strip(), l.split('='))) for l in cf if l[0] != '#']}

    f.close()
    del cf, config_file, f

CACHE_DIR = os.path.join(str(Path.home()), '.gscapl')
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def dpath(x):
    return os.path.join(CACHE_DIR, x)


zname = os.path.join(__file__.replace('utils.py', ''), 'zips.txt')
zips = pd.read_csv(zname)
zips = zips.set_index('zipcode')
ztree = KDTree(zips[['lat', 'lon']].values)
del zname


def isint(x):
    """Determine if provided object is convertible to an int

    Args:
        x: object

    Returns:
        bool
    """
    if x is None:
        return False

    try:
        int(float(x))
        return True
    except:
        return False


def isfloat(x):
    """Determine if provided object is convertible to a float

    Args:
        x: object

    Returns:
        bool
    """
    if x is None:
        return False

    try:
        float(x)
        return True
    except:
        return False


def dd_from_zip(zipcode):
    """Get the latitude and longitude coordinate pair for the center of the provided zipcode

    Args:
        zipcode: (int | str)

    Returns:
        (float, float)
    """
    try:
        zipcode = check_zipcode_type(zipcode)

        lat = zips.loc[zipcode].lat.values[0]
        lon = zips.loc[zipcode].lon.values[0]
        return lat, lon
    except:
        return 0, 0


def zip_from_dd(lat, lon, maxd=sys.maxsize, suppress_warnings=False):
    """Get the closest zipcode to a latitude, longitude coordinate pair

    Args:
        lat: float - Latitude in degree-decimal (DD) format
        lon: float - Longitude in degree-decimal (DD) format
        maxd: (optional) float - Maximum distance in kilometers for which to return a result
        suppress_warnings: (optional) bool - set to True to suppress distance to zipcode warnings

    Returns:
        int - the zipcode found or -1 if not

    Raises:
        TypeError if and of lat, lon, or maxd is not an int or float
    """
    if not isinstance(lat, (float, int)) or not isinstance(lon, (float, int)) or not isinstance(maxd, (float, int)):
        raise TypeError('lat, lon and maxdistance must be ints or floats')

    lat_lon_range_check(lat, lon)

    try:
        nearest = ztree.query(
            (lat, lon),
            k=1,
            distance_upper_bound=maxd
        )
        if not suppress_warnings:
            if nearest[0] == float('inf'):
                print(f'WARNING: zipcode not found within {maxd}Km of ({lat}, {lon})')
                return -1
            elif nearest[0] > 100:
                print(f'WARNING: closest zipcode found was {np.round(nearest[0], 1)}Km from ({lat}, {lon})')
            else:
                pass
        else:
            pass

        return int(zips.iloc[nearest[1]].name)
    except:
        return -1


def tz_from_dd(points):
    """Get the timezone for a coordinate pair

    Args:
        points: (lat, lon) | [(lat, lon),] | pd.DataFrame w/lat and lon as columns

    Returns:
        np.array
    """
    if isinstance(points, pd.DataFrame):
        points = points.values.tolist()

    if not isinstance(points, list):
        points = [points]

    x = ztree.query(points)
    x = zips.iloc[x[1]].timezone.values
    return x


def tz_from_zip(zipcode):
    """Get the timezone from a zipcode

    Args:
        zipcode: str|int | [str|int,] | pd.Series

    Returns:
        np.array
    """
    if isinstance(zipcode, pd.Series):
        zipcode = zipcode.tolist()

    if not isinstance(zipcode, list):
        zipcode = [zipcode]
    else:
        pass

    points = [dd_from_zip(zc) for zc in zipcode]
    return tz_from_dd(points)


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


def check_zipcode_type(zipcode):
    if not isinstance(zipcode, (str, int, float)) or \
            isinstance(zipcode, str) and not isint(zipcode):
        raise TypeError
    elif isinstance(zipcode, (str, float)):
        zipcode = int(float(zipcode))
    else:
        pass

    if zipcode < 0:
        raise ValueError

    return zipcode


def lat_lon_range_check(lat, lon):
    if -90 > lat or lat > 90:
        raise ValueError('Latitude must be in valid range: -90 < lat < 90.')

    if -180 > lon or lon > 180:
        raise ValueError('Longitude must be in valid range: -180 < lon < 180.')


if __name__ == '__main__':
    pass
