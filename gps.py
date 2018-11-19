#!/usr/bin/env python

""" A collection of scripts for processing GPS streams"""

import ast
from collections import *
from contextlib import contextmanager
import datetime as dt
from enum import Enum
import json
import multiprocessing as mul
import os
from pathlib import Path
import re
import requests
from sqlite3 import dbapi2 as sqlite
import time
from urllib3.exceptions import ConnectionError

import googlemaps
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sqlalchemy import and_, create_engine
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
import synapseclient

__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'

"""How many times to retry a network timeout 
and how many seconds to wait between each """
CONNECTION_RESET_ATTEMPTS = 99
CONNECTION_WAIT_TIME = 60

"""named tuple used to clarify types used in the scripts below"""
GPS = namedtuple('GPS', ['lat', 'lon', 'ts'])
Cluster = namedtuple('Cluster', ['lat', 'lon', 'cid'])


@contextmanager
def synapse_scope():
    syn = synapseclient.Synapse()
    syn.login()

    try:
        yield syn
    except Exception as e:
        raise e
    finally:
        syn.logout()


"""verify db cache exists or download if necessary"""
dpath = lambda x: os.path.join(Path.home(), '.gscap', x)

if not os.path.exists(dpath('')):
    os.mkdir(dpath(''))

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

"""only open this zips once, download if unavailable"""
zname = 'zips.csv'
if not os.path.exists(dpath(zname)):
    with synapse_scope() as s:
        zips = s.tableQuery('SELECT zipcode, lat, lon FROM syn16816780').asDataFrame()
        zips.to_csv(dpath(zname), index=None)

zips = pd.read_csv(dpath(zname))
zips = zips.set_index('zipcode')
del zname


# --------------------------------------------------------------------------
# Misc stuff
# --------------------------------------------------------------------------
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


def isnum(x):
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


# --------------------------------------------------------------------------
# API calls and caching
# --------------------------------------------------------------------------
""" SQLAlchemy declarative base for ORM features """
Base = declarative_base()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except IntegrityError:
        session.rollback()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


class PlaceRequest(Base):
    __tablename__ = 'apicalls'

    lat = Column(Float, primary_key=True)
    lon = Column(Float, primary_key=True)
    source = Column(String, primary_key=True)
    radius = Column(Float,  primary_key=True)
    rankby = Column(String, primary_key=True)
    dtRetrieved = Column(DateTime)
    content = Column(String)

    def __init__(self, lat=None, lon=None, radius=None, rankby=None, source=None):
        self.lat = np.round(lat, 5) if isnum(lat) else np.nan
        self.lon = np.round(lon, 5) if isnum(lon) else np.nan
        self.valid = self.__verify_location()

        self.radius = radius
        self.rankby = rankby.value

        if source is not None:
            self.source = source.value.get('name')
            self.call = source.value.get('call')
            self.parse_content = source.value.get('parse_content')

    def make_call(self):
        pass

    @property
    def dict(self):
        return dict(
            lat=self.lat,
            lon=self.lon,
            radius=self.radius,
            source=self.source,
            dtRetrieved=dt.datetime.now(),
            content=self.content
        )

    @property
    def dataframe(self):
        return pd.DataFrame(self.dict, index=[0])

    def from_tuple(self, t, source):
        self.dtRetrieved  = t.dtRetrieved
        self.lat = t.lat
        self.lon = t.lon
        self.radius  = t.radius
        self.source  = t.source
        self.rankby  = t.rankby
        self.content = t.content

        self.call  = source.value['call']
        self.categories = source.value['categories']
        return self

    def __verify_location(self):
        lat, lon = self.lat, self.lon

        if lat is None and lon is None:
            return False

        elif (lat is not None and lon is not None) and \
                (lat < -90 or lat > 90 or lon < -180 or lon > 180):
            raise ValueError('lat, lon must be in a valid range')

        else:
            pass

        self.lat = lat
        self.lon = lon

        return lat != 0 and lon != 0

    def __repr__(self):
        return f'<PlaceRequest(lat={self.lat}, lon={self.lon}, source={self.source})>'


"""ensure the api cache is ready"""
gcname = 'api_cache.sqlite'
engine = create_engine(f'sqlite+pysqlite:///{dpath(gcname)}', module=sqlite)
Base.metadata.create_all(engine)
del gcname


# Yelp -----------------------------------------------------------------
class YelpRankBy(Enum):
    BEST_MATCH = 'best_match'
    RATING = 'rating'
    REVIEW_COUNT = 'review_count'
    DISTANCE = 'distance'


with synapse_scope() as s:
    YELP_TYPE_MAP = pd.read_csv(s.get('syn17011507').path).set_index('cat')


def yelp_call(request):
    try:
        key = CONFIG['YelpAPI']
    except KeyError:
        print('a key for YelpAPI was not found in .gscapConfig file')

    url = 'https://api.yelp.com/v3/businesses/search'
    params = {
        'latitude': request.lat,
        'longitude': request.lon,
        'radius': request.radius,
        'sort_by': request.rankby
    }
    complete = False

    while not complete:
        with requests.Session() as s:
            s.headers.update({
                'Authorization': f'Bearer {key}'
            })

            response = s.get(url, params=params)
            if response.ok:
                result = response.json()
                complete = True
            else:
                time.sleep(np.random.uniform(.01, 5, 1)[0])

        if 'TOO_MANY_REQUESTS_PER_SECOND' in str(response.content):
            time.sleep(np.random.uniform(.01, 5, 1)[0])
            complete = False

    request.content = json.dumps(result)
    return request


def parse_yelp_response(c):
    if not isinstance(c, str):
        raise TypeError('content must be a string')

    empty = dict(
            name='not found',
            rank_order=-1,
            categories='none',
            major_categories='none'
        )

    if c is not None and c.lower() != 'nan':
        results = None

        try:
            c = json.loads(c)
        except json.JSONDecodeError as e:
            return dict(
                name=str(e),
                rank_order=-1,
                categories=c,
                major_categories='JSONDecodeError'
            )

        businesses = c.get('businesses')
        if businesses is not None:
            for i, r in enumerate(businesses):
                name = r.get('name')
                minor = [ri.get('alias') for ri in r.get('categories')]
                major = list(set([YELP_TYPE_MAP.loc[mi, 'mapping'] for mi in minor]))

                if 'dining_out' in major:
                    major = ['dining_out']

                if len(major) > 1:
                    major = [major[0]]

                results = dict(
                    name=name,
                    rank_order=i,
                    categories=', '.join(minor),
                    major_categories=', '.join(major)
                )
                break
        return results or empty
    else:
        return empty


# GMAPS ----------------------------------------------------------------
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


class GmapsRankBy(Enum):
    PROMINENCE = 'prominence'


with synapse_scope() as s:
    GMAP_TYPE_MAPPINGS = pd.read_csv(s.get('syn17011508').path).set_index('cat')


def gmapping(x):
    t = None
    try:
        t = GMAP_TYPE_MAPPINGS.loc[x].mapping
    except KeyError:
        pass

    if isinstance(t, pd.Series):
        t = t.tolist()[0]

    if t is not None and 'Expecting value:' in t:
        t = 'JSON Decode Error'

    return {t} if t is not None else {'undefined category'}


def gmap_call(request):
    try:
        key = CONFIG['GooglePlacesAPI']
    except KeyError:
        print('a key for GooglePlacesAPI was not found in .gscapConfig file')

    gmaps = googlemaps.Client(key=key)
    nearby_request = gmaps.places_nearby(
        location=(request.lat, request.lon),
        radius=request.radius,
        rank_by=request.rankby
    )

    request.content = json.dumps(nearby_request)
    return request


def parse_gmap_response(c):
    if c is not None:
        ci_complete = False
        results = None

        try:
            # remove dom hyperlinks
            c = re.sub(r'</?a[^>]*?>', '', c)
            c = json.loads(c)
        except json.JSONDecodeError as e:
            return dict(
                rank_order=-1,
                name=str(e),
                categories=c,
                major_categories='JSONDecodeError'
            )

        for i, r in enumerate(c.get('results')):
            types = set(r.get('types'))

            if any([t in IGNORED_PLACE_TYPES for t in types]) or ci_complete:
                continue

            else:
                name = r.get('name')

                # remove ambiguous types
                types = types - {'point_of_interest', 'establishment', 'premise'}

                # pull out the major categories
                major = {
                    'food', 'store', 'repair', 'finance',
                    'restaurant', 'park', 'health', 'transit_station',
                    'lodging', 'place_of_worship', 'doctor'
                }
                mc = types.intersection(major)
                mc = mc if len(mc) > 0 else {'other'}
                types = types - major

                # manually set type by known names
                if name in ['Sears', 'Macy\'s', 'mygofer', 'Target', 'T.J. Maxx']:
                    types = {'department_store'}

                elif name in ['Fred Meyer']:
                    types = {'supermarket'}

                # manually reduce
                elif 'gas_station' in types:
                    types = {'gas_station'}

                elif 'lodging' in mc:
                    types = {'lodging'}

                elif 'transit_station' in mc:
                    types = {'transit_station'}

                elif mc == {'health', 'doctor'} or \
                        mc == {'store', 'health', 'doctor'}:
                    types = {'health'}

                elif 'health' in mc and 'store' in mc:
                    types = {'supermarket'}

                elif mc == {'store', 'finance'}:
                    types = {'finance'}

                elif mc == {'store', 'general_contractor'}:
                    types = {'repair'}

                elif 'restaurant' in mc:
                    mc = gmapping('restaurant')

                elif mc == {'food', 'store'}:
                    mc = gmapping('supermarket')

                elif mc == {'food', 'store', 'general_contractor'}:
                    types = {'consumer_goods'}

                # take the left most
                if len(types) == 0:
                    types = mc
                elif len(types) == 1:
                    mc = gmapping(list(types)[0])
                elif len(types) > 1:
                    t = list(types)[0]
                    types = {t}
                    mc = gmapping(t)

                if len(mc) > 1:
                    mc = {list(mc)[0]}

                if mc == {'store'}:
                    mc = gmapping('store')

                elif mc == {'food'}:
                    mc = gmapping('food')

                if mc == {'other'} and len(types) == 0:
                    types = {'other'}

                # or take the major if type doesn't remain
                results = dict(
                    rank_order=i,
                    name=name,
                    categories=', '.join(types),
                    major_categories=', '.join(mc)
                )

                ci_complete = True

        return results or dict(rank_order=-1, name='not found', categories='none', major_categories='none')


# processing ------------------------------------------------------------
def process_request(args):
    request, force, progress_qu, request_qu, response_qu = args

    if not request.valid:
        return request.dataframe

    my_pid = os.getpid()
    request_qu.put(dict(pid=my_pid, type='get', args=(request,)))
    r = response_qu.get()

    if not force:
        while r['pid'] != my_pid:
            response_qu.put(r)
            r = response_qu.get()

        if r['response'] is not None:
            if progress_qu is not None:
                progress_qu.put(1)
            else:
                pass

            return dict(report=r['response'].dict, hits=1, misses=0)
        else:
            pass

    # if we made it this far, we had a cache miss so make the request
    a, call_complete = 0, False
    while a < CONNECTION_RESET_ATTEMPTS and not call_complete:
        try:
            result = request.call(request)
            call_complete = True
        except ConnectionError:
            a += 1
            print(f'\n{a} connection attempts failed\n')
            time.sleep(CONNECTION_WAIT_TIME)
        except Exception as e:
            raise e

    # cache our results
    request_qu.put(dict(type='put', args=(result,)))

    # if monitoring using the progress_bar in utils and a qu is supplied
    if progress_qu is not None:
        progress_qu.put(1)
    else:
        pass

    return dict(report=result.dict, hits=0, misses=1)


def request_nearby_places(request, n_jobs=1, force=False, progress_qu=None):
    if not isinstance(request, list):
        request = [request]

    # first check our cache to see of we've made the same request
    man = mul.Manager()
    request_qu, response_qu = man.Queue(), man.Queue()
    cman = mul.Process(
        target=cache_manager, args=(request_qu, response_qu)
    )
    cman.start()

    cpus = mul.cpu_count()
    cpus = cpus if n_jobs == -1 or n_jobs >= cpus else n_jobs
    pool = mul.Pool(cpus)

    if len(request) == 1:
        results = [
            process_request((request[0], force, progress_qu, request_qu, response_qu))
        ]
        request_qu.put(dict(type='end'))
    else:
        results = list(pool.map(
             process_request,
             [(ri, force, progress_qu, request_qu, response_qu) for ri in request]
        ))
        request_qu.put(dict(type='end'))

    pool.terminate(); pool.join(); del pool
    cman.terminate(); cman.join(); del cman
    man.shutdown(); del man

    hits = np.sum([r['hits'] for r in results])
    misses = np.sum([r['misses'] for r in results])
    results = pd.DataFrame(
        [r['report'] for r in results],
        index=np.arange(len(results))
    )

    for r, ir in zip(request, results.iterrows()):
        idx, row = ir

        t = r.parse_content(str(row.content))
        results.loc[idx, 'name'] = t['name']
        results.loc[idx, 'rank_order'] = t['rank_order']
        results.loc[idx, 'categories'] = t['categories']
        results.loc[idx, 'major_categories'] = t['major_categories']

    results = results.drop(columns='content')

    return dict(request=results, hits=hits, misses=misses)


def cache_manager(request_qu, response_qu):
    finished = False

    while not finished:
        with session_scope() as session:
            try:
                r = request_qu.get()
            except EOFError:
                finished = True
                continue

            if r['type'] == 'get':
                request = r['args'][0]
                content = get_from_cache(request, session)
                response_qu.put(
                    dict(
                        pid=r['pid'],
                        response=content)
                )

            elif r['type'] == 'put':
                put_to_cache(r['args'][0], session)

            elif r['type'] == 'end':
                finished = True


def get_from_cache(r, session):
    a = session.query(PlaceRequest).filter(and_(
        (PlaceRequest.lat == r.lat),
        (PlaceRequest.lon == r.lon),
        (PlaceRequest.radius == r.radius),
        (PlaceRequest.source == r.source)
    )).first()

    return  a


def put_to_cache(r, session):
    a = session.query(PlaceRequest).filter(and_(
        (PlaceRequest.lat == r.lat) &
        (PlaceRequest.lon == r.lon) &
        (PlaceRequest.source == r.source) &
        (PlaceRequest.radius == r.radius)
    )).all()

    for ai in a:
        ai.delete()

    now = dt.datetime.now()
    r.dtRetrieved = dt.datetime(
        year=now.year, month=now.month, day=now.day, hour=now.hour, minute=now.minute
    )
    session.add(r)


# source mappings ------------------------------------------------------
def api_source(item):
    if item == 'Google Places':
        return ApiSource.GMAPS

    elif item == 'Yelp':
        return ApiSource.YELP

    else:
        raise KeyError(f'{item} is not a valid API source')


class ApiSource(Enum):
    YELP = dict(
        name='Yelp',
        call=yelp_call,
        parse_content=parse_yelp_response
    )
    GMAPS = dict(
        name='Google Places',
        call=gmap_call,
        parse_content=parse_gmap_response
    )
    FROM_NAME = api_source


# --------------------------------------------------------------------------
# GPS and clustering
# --------------------------------------------------------------------------
def cluster_metrics(clusters, entries):
    if 'cnt' in clusters.columns:
        clusters.drop(columns='cnt', inplace=True)
    else:
        pass

    stats = []
    grouped = entries.groupby('cid')

    for name, group in grouped:
        df = pd.DataFrame(group)
        df = df.sort_values(by='midpoint')

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

        # mean time interval between visits - Mobile Phone Detection of Semantic Location and...
        # rolling mean between entry midpoint pairs
        df['ns'] = df.midpoint.apply(lambda x: x.timestamp())
        mti = df.ns.rolling(window=2).apply(lambda x: x[1] - x[0], raw=True)
        mti = np.round(np.mean(mti)/3600, 3)

        stats.append(
            dict(
                cid=name,
                times_entered=len(df),
                total_duration=np.round(df.duration.sum().total_seconds()/3600, 3),
                mean_duration=np.round(df.duration.mean().total_seconds()/3600, 3),
                std_duration=np.round(df.duration.std().total_seconds()/3600, 3),
                max_duration=np.round(df.duration.max().total_seconds()/3600, 3),
                min_duration=np.round(df.duration.min().total_seconds()/3600, 3),
                # earliest_time_in=df.in_tod.min(),
                # latest_time_in=df.in_tod.max(),
                # earliest_time_out=df.out_tod.min(),
                # latest_time_out=df.out_tod.max(),
                # days_entered=int(len(daily_counts)),
                # count_before_9=int(np.sum([
                #     1 for r in df.in_tod if r < dt.time(hour=9)
                # ])),
                # count_between_9_12=int(np.sum([
                #     1 for r in df.in_tod
                #     if dt.time(hour=9) <= r < dt.time(hour=12)
                # ])),
                # count_between_12_5=int(np.sum([
                #     1 for r in df.in_tod
                #     if dt.time(hour=5) <= r < dt.time(hour=17)
                # ])),
                # count_after_5=int(np.sum([
                #     1 for r in df.in_tod if dt.time(hour=17) <= r
                # ])),
                # max_entries_per_day=int(np.max(daily_counts)),
                # min_entries_per_day=int(np.min(daily_counts)),
                # mean_entries_per_day=np.round(np.mean(daily_counts), 3),
                # std_entries_per_day=np.round(np.std(daily_counts), 3),
                # max_entries_per_week=int(np.max(weekly_counts)),
                # min_entries_per_week=int(np.min(weekly_counts)),
                # mean_entries_per_week=np.round(np.mean(weekly_counts), 3),
                # std_entries_per_week=np.round(np.std(weekly_counts), 3),
                # max_entries_per_month=int(np.max(ymonthly_counts)),
                # min_entries_per_month=int(np.min(ymonthly_counts)),
                # mean_entries_per_month=np.round(np.mean(ymonthly_counts), 3),
                # std_entries_per_month=np.round(np.std(ymonthly_counts), 3),
                # most_common_hod_in=int(counts_by_itod[0][0]),
                # most_common_hod_out=int(counts_by_otod[0][0]),
                # most_common_dow=int(counts_by_dow[0][0]),
                # most_common_dow_cnt=int(counts_by_dow[0][1]),
                # most_common_month=int(counts_by_month[0][0]),
                # most_common_month_cnt=int(counts_by_month[0][1]),
                mean_ti_between_visits=mti,
            )
        )

    if len(stats) > 0:
        stats = pd.DataFrame().from_dict(stats)
        stats.set_index('cid', inplace=True)

        clusters = clusters.join(stats, on='cid', how='outer', sort=True)
        return clusters
    else:
        return None


def process_velocities(records):
    cols = ['lat', 'lon', 'ts']

    if isinstance(records, list):
        records = pd.DataFrame(records, columns=cols)

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
    records = pd.concat([records, x], axis=1, sort=True)
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
        available bins: { 'stationary', 'walking', 'active', 'powered_vehicle', 'high_speed_transportation', 'anomaly' }
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
        binning = 'active'

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
        if 0 < g.ts.hour < 6  # change made by recomendation of paper.
        # TODO: FIND THE SOURCE
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
        # idx = [i for i, k in enumerate(dbscan.labels_) if k == ci]
        idx = np.where(dbscan.labels_ == ci)[0]

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


def geo_pairwise_distances(x, as_array=True, n_jobs=1):
    """

    Args:
        x: [(float, float),]
        n_jobs: number of cores to use, -1 for all

    Returns:
        [float,]
    """
    idx_pairs = [(i, j) for i in range(len(x)) for j in range(i+1, len(x))]
    pairs = [(x[p[0]], x[p[1]]) for p in idx_pairs]

    cpus = mul.cpu_count()
    cpus = cpus if n_jobs <= 0 or n_jobs >= cpus else n_jobs
    pool = mul.Pool(cpus)

    result = list(pool.map(__geo_distance_wrapper, pairs))
    result = np.round(result, 1)

    x=pd.DataFrame(
        [
            (p[0][0], p[0][1], p[1][0], p[1][1], d)
            for p, d in zip(pairs, result)
        ],
        columns=['lat1', 'lon1', 'lat2', 'lon2', 'distance']
    )

    pool.close()
    pool.join()
    return np.array(sorted(result)) if as_array else x


def get_clusters_with_context(records, parameters=None, validation_metrics=False):
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
    if 'working' not in records.columns or len(records.loc[records.working == True]) > 0:
        work, wmask = estimate_work_location(stationary, parameters)
        work_records = stationary.iloc[wmask, :].copy()
        work_records.cid = 'work'
    else:
        work = work_records = None
        wmask = []

    # scan the remaining records
    rmask = list(set(np.arange(stationary.shape[0])) - set(hmask+wmask))
    remaining = stationary.iloc[rmask, :].copy()
    gps_records = [
        GPS(lat=r.lat, lon=r.lon, ts=r.ts) for r in remaining.itertuples()
    ]
    labels, clusters = gps_dbscan(gps_records, parameters)

    remaining.cid = [f'x{l}' if l != -1 else 'xNot' for l in labels]

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

    records = pd.concat([home_records, work_records, remaining, others], sort=True)

    clusters['name'] = 'nap'
    clusters['categories'] = 'nap'
    clusters.loc[clusters.cid == 'home', 'name'] = 'home'
    clusters.loc[clusters.cid == 'home', 'categories'] = 'home'
    clusters.loc[clusters.cid == 'work', 'name'] = 'work'
    clusters.loc[clusters.cid == 'work', 'categories'] = 'work'

    if not validation_metrics:
        clusters = clusters.drop(columns=[
            'lat_IQR', 'lat_max', 'lat_min', 'lat_range', 'lat_std',
            'lon_IQR', 'lon_max', 'lon_min', 'lon_range', 'lon_std',
            'max_distance_from_center'
        ])

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

    def ccat(cid):
        c = clusters.loc[clusters.cid == cid, 'categories'].values

        if len(c) > 0:
            return c[0]

        return 'err'

    if 'cid' not in entries.columns:
        return None
    else:
        entries['cname'] = entries.cid.apply(cname)
        entries['category'] = entries.cid.apply(ccat)

    return entries


def get_daily_metrics(records, entries):
    t = []
    records['day'] = records.ts.apply(lambda x: x.date())
    days = pd.unique(records.day)

    top_3 = list(
        records.loc[
             (records.cid != 'home') &
             (records.cid != 'work') & (records.cid != 'xNot'),
             ['cid', 'time_delta']
        ].groupby('cid').sum().reset_index().sort_values('time_delta', ascending=False).iloc[:3].cid.values
    )

    ln_sleep = None
    for day in days:
        # get the days records
        de = entries.loc[entries.date == day]
        r = records.loc[records.day == day]

        # quantify hours of sleep using only if last night's ending cluster was the same
        # as the morning and was in fact the day prior
        sleep = pd.to_datetime(r.loc[r.ts == np.min(r.ts)].ts.values)
        sleep = sleep[0] if len(sleep) > 0 else None

        c2 = r.loc[r.ts == np.min(r.ts)].cid.values
        c2 = c2[0] if len(c2) > 0 else None

        if sleep is not None and c2 is not None:
            sleep = dt.datetime(
                year=sleep.year, month=sleep.month, day=sleep.day, hour=sleep.hour, minute=sleep.minute
            ) - dt.datetime(year=sleep.year, month=sleep.month, day=sleep.day, hour=0, minute=0)

            if ln_sleep is not None and (sleep + ln_sleep[0]).total_seconds() / 3600 < 24 and c2 == c1:
                sleep = (ln_sleep[0] + sleep).total_seconds() / 3600
            else:
                sleep = np.nan
        else:
            sleep = np.nan

        # calculate the time spent in the subject's overall top 3 cluster
        todays_clusters = pd.unique(de.cid)
        ct = [
            de.loc[de.cid == ci].duration.sum().total_seconds()
            for ci in todays_clusters if ci in top_3]
        ct = np.sum(ct)

        # calculate the time spent in the home and work clusters
        home = de.loc[de.cid == 'home'].duration.sum().total_seconds() / 3600
        work = de.loc[de.cid == 'work'].duration.sum().total_seconds() / 3600

        # calculate how many hours in the day had GPS data
        hod = len(pd.unique(r.ts.apply(lambda r: r.hour)))

        # stationary hours outside home/work clusters
        non_hw_stationary = np.round(
            r.loc[
                (r.cid != 'home') & (r.cid != 'work') & (r.binning == 'stationary')
         ].time_delta.sum() / 3600, 3)

        # location variance - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5571235/
        lv = np.std(r.lat) + np.std(r.lon)
        lv = np.log(lv) if lv > 0 else np.nan

        dm = dict(
            date=day,
            location_variance=lv,
            hours_accounted_for=hod,
            hours_without_data=24 - hod,
            hours_stationary=np.round(r.loc[r.binning == 'stationary'].time_delta.sum() / 3600, 3),
            hours_stationary_non_home_work=non_hw_stationary,
            hours_walking=np.round(r.loc[r.binning == 'walking'].time_delta.sum() / 3600, 3),
            distance_walking=np.round(r.loc[r.binning == 'walking'].displacement.sum()),
            hours_active=np.round(r.loc[r.binning == 'active'].time_delta.sum() / 3600, 3),
            distance_active=np.round(r.loc[r.binning == 'active'].displacement.sum()),
            hours_powered_vehicle=np.round(r.loc[r.binning == 'powered_vehicle'].time_delta.sum() / 3600, 3),
            distance_powered_vehicle=np.round(r.loc[r.binning == 'powered_vehicle'].displacement.sum(), 3),
            hours_high_speed_transportation=np.round(
                r.loc[r.binning == 'high_speed_transportation'].time_delta.sum() / 3600, 3),
            distance_high_speed_transportation=np.round(
                r.loc[r.binning == 'high_speed_transportation'].displacement.sum(), 3),
            hours_of_sleep=np.round(sleep, 3) if sleep is not np.nan else np.nan,
            hours_at_home=np.round(home, 3),
            hours_at_work=np.round(work, 3),
            came_to_work=any(de.cid == 'work'),
            number_of_clusters=len(todays_clusters),
            hours_spent_in_top_3_clusters=np.round(ct / 3600, 3)
        )

        t.append(dm)

        # create some metrics for the next pass to calculate sleep patterns
        t1 = r.loc[r.ts == np.max(r.ts)].ts
        t1 = t1.iloc[0] if len(t1) > 0 else None

        c1 = r.loc[r.ts == np.max(r.ts)].cid.values
        c1 = c1[0] if len(c1) > 0 else None

        if t1 is not None and c1 is not None:
            t1 = dt.datetime(year=t1.year, month=t1.month, day=t1.day, hour=t1.hour, minute=t1.minute)
            ln_sleep = ((dt.datetime(year=t1.year, month=t1.month, day=t1.day, hour=23, minute=59) - t1),
                        t1, c1)
        else:
            ln_sleep = None

    return pd.DataFrame(t)


def get_next_phase_clusters(records, clusters, params, min_distance=100, validation_metrics=False):
    def getint(ri):
        return int(ri[1:]) if isnum(ri[1:]) else None

    # add an exclusion mask to identify which points to perform the clustering
    records['exmask'] = ~((records.cid != 'xNot') | (records.binning != 'stationary'))
    records['day'] = records.ts.apply(lambda x: x.date())

    next_cid = [rj for rj in [getint(ri) for ri in records.cid] if rj is not None]
    next_cid = np.max(next_cid) + 1 if len(next_cid) > 0 else 0

    cur_cluster_set = clusters.loc[:, ['lat', 'lon']]

    cs = []
    for day in pd.unique(records.day):
        records['dmask'] = records.day == day

        # only include points more than x meters away from an existing cluster
        for idx, row in records.loc[records.exmask & records.dmask].iterrows():
            if any([
                geo_distance(row.lat, row.lon, c.lat, c.lon) < min_distance
                for c in cur_cluster_set.itertuples()
            ]):
                records.loc[idx, 'dmask'] ^= True

        assert all(
            [cid == 'xNot' for cid in records.loc[records.exmask & records.dmask].cid]
        )

        # build record set and push through dbscan
        gpsr = [
            GPS(lat=g.lat, lon=g.lon, ts=g.ts)
            for g in records.loc[records.exmask & records.dmask].itertuples()
        ]
        labels, dcs = gps_dbscan(gpsr, params)

        # reset the cids to start where phase 1 left off
        if len(dcs) > 0:
            cids = [f'x{next_cid+l}' if l != -1 else 'xNot' for l in labels]
            assert len(cids) == len(records.loc[records.exmask & records.dmask]['cid'])

            records.loc[records.exmask & records.dmask, 'cid'] = cids

            # convert to dataframe and reset cluster id's to begin where phase 1 left off
            dcs = pd.DataFrame(dcs)
            dcs.cid = [f'x{cid+next_cid}' for cid in dcs.cid]
            cs.append(dcs)

            next_cid += len(dcs)

    if len(cs) > 0:
        cs = pd.concat(cs, sort=True)
        cs['categories'] = cs['name'] = 'nap'

        clusters = pd.concat([clusters, cs], sort=False)

        if not validation_metrics:
            clusters = clusters.drop(columns=[
                'lat_IQR', 'lat_max', 'lat_min', 'lat_range', 'lat_std',
                'lon_IQR', 'lon_max', 'lon_min', 'lon_range', 'lon_std',
                'max_distance_from_center'
            ])

    records = records.drop(columns=['exmask', 'day'])
    return records, clusters


def gps_dbscan(gps_records, parameters=None):
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

    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs
    ).fit([(g.lat, g.lon) for g in gps_records])
    assert len(gps_records) == len(dbscan.labels_)

    clusters = extract_cluster_centers(gps_records, dbscan)
    assert len(set(pd.unique(dbscan.labels_)) - {-1}) == len(clusters)

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

    b, d, sec = metrics['binning'], metrics['displacement'], metrics['time_delta']
    if b != 'stationary' or d > 75 or sec > 60**2*12:
        return None

    a_lat, a_lon, a_ts = coordinate_a
    b_lat, b_lon, b_ts = coordinate_b

    if not (isinstance(a_ts, dt.datetime) and isinstance(b_ts, dt.datetime)):
        raise TypeError('third element of each coordinate tuple must be dt')

    fill_range = list(pd.date_range(a_ts, b_ts, freq=freq))

    # ensure the returned dataframe range is exclusive
    if fill_range[0] == a_ts:
        fill_range.remove(fill_range[0])

    if len(fill_range) == 0:
        return

    if fill_range[-1] == b_ts:
        fill_range.remove(fill_range[-1])

    fill_lat = np.linspace(a_lat, b_lat, len(fill_range))
    fill_lon = np.linspace(a_lon, b_lon, len(fill_range))

    t = dict(lat=fill_lat, lon=fill_lon, ts=fill_range)
    return pd.DataFrame(t)


def impute_stationary_coordinates(records, freq='5Min', metrics=True):
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
            x = pd.concat(x, sort=True)
            records = pd.concat([records, x], axis=0, sort=True)
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
        records = pd.concat(all_days, axis=0, sort=True)
        records.sort_values('ts', inplace=True)
        records.index = np.arange(len(records))

    # cleanup
    if 'day' in records.columns:
        records.drop(columns='day', inplace=True)
    else:
        pass

    if not metrics:
        records.drop(
            columns=['binning', 'displacement', 'time_delta', 'velocity'],
            inplace=True
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


def __geo_distance_wrapper(args):
    a, b = args
    return geo_distance(*a, *b)


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


if __name__ == '__main__':
    pass
