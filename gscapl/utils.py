#!/usr/bin/env python

""" A collection of data processing scripts """

import datetime as dt
from multiprocessing import Manager, Queue, Process
import re
from pathlib import Path
import os

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


zname = os.path.join(__file__.replace('utils.py', ''), 'data', 'zips.csv')
zips = pd.read_csv(zname)
zips = zips.set_index('zipcode')
ztree = KDTree(zips[['lat', 'lon']].values)
del zname


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


def isint(x):
    if x is None:
        return False

    try:
        int(float(x))
        return True
    except ValueError:
        return False


def isfloat(x):
    if x is None:
        return False

    try:
        float(x)
        return True
    except ValueError:
        return False


def dd_from_zip(zipcode):
    try:
        lat = zips.loc[zipcode].lat.values[0]
        lon = zips.loc[zipcode].lon.values[0]
        return lat, lon
    except:
        return 0, 0


def zip_from_dd(lat, lon):
    try:
        # get closest zip within 7 miles
        win68miles = zips.loc[
            (np.round(zips.lat, 0) == np.round(lat, 0)) &
            (np.round(zips.lon, 0) == np.round(lon, 0))
        ].dropna()

        win68miles['d'] = [geo_distance(lat, lon, r.lat, r.lon) for r in win68miles.itertuples()]
        return zips.loc[win68miles.d.idxmin()].index.tolist()[0]
    except:
        return -1


def tz_from_dd(points):
    x = ztree.query(points)
    x = zips.iloc[x[1]].timezone.values
    return x


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


def progress_bar(pbar):
    """progress bar to track parallel events
    Args:
        total: (int) total number of tasks to complete
        desc: (str) optional title to progress bar
    Returns:
        (Process, Queue)
    """
    proc_manager = Manager()

    def track_it(pbar, trackq):
        idx = 0

        while True:
            try:
                update = trackq.get()

                if update is None:
                    break

            except EOFError:

                break

            pbar.update(update)
            idx += 1

    trackq = proc_manager.Queue()
    p = Process(target=track_it, args=(pbar, trackq))
    p.start()

    return p, trackq


def pcmap(func, vals, n_cons, pbar=False, **kwargs):
    """parallel mapping function into producer-consumer pattern
    Args:
        func: (Function) function to apply
        vals: [Object] values to apply
        n_cons: (int) number of consumers to start
        pbar: (bool)

    Returns:
        [Object] list of mapped function return values
    """
    if pbar:
        total = len(vals)
        desc = kwargs.get('desc')

        pbar, trac_qu = progress_bar(total, desc)
    else:
        pbar, trac_qu = None, None

    def consumer(c_qu, r_qu, func):
        """consumer, terminate on receiving 'END' flag
        Args:
            c_qu: (Queue) consumption queue
            r_qu: (Queue) results queue
        """
        while True:
            val = c_qu.get()

            if isinstance(val, str) and val == 'END':
                break

            rv = func(val)
            r_qu.put(rv)

        r_qu.put('END')

    # create queues to pass tasks and results
    consumption_queue = Queue()
    results_queue = Queue()

    # setup the consumers
    consumers = [
        Process(target=consumer, args=(
            consumption_queue,
            results_queue,
            func
        ))
        for i in range(n_cons)
    ]

    # start the consumption processes
    [c.start() for c in consumers]

    # dish out tasks and add the termination flag
    [consumption_queue.put(val) for val in vals]
    [consumption_queue.put('END') for c in consumers]

    # turn the results into a list
    running, brake, results = n_cons, False, []
    while not brake:
        while not results_queue.empty():
            val = results_queue.get()

            if isinstance(val, str) and val == 'END':
                running -= 1

                if running == 0:
                    brake = True
            else:
                if trac_qu is not None:
                    trac_qu.put(1)
                else:
                    pass

                results.append(val)

    # kill and delete all consumers
    [c.terminate() for c in consumers]
    del consumers

    # kill the progress bar
    if pbar is not None:
        pbar.terminate()
        pbar.join()
        del pbar, trac_qu
    else:
        pass

    return results


if __name__ == '__main__':
    pass
