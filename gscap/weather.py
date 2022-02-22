#!/usr/bin/env python

""" A collection of scripts for gathering weather data """
from collections import namedtuple
from collections import namedtuple
from contextlib import contextmanager
import datetime as dt
import multiprocessing as mul
import requests
from requests.exceptions import ConnectionError
from sqlite3 import dbapi2 as sqlite
import time

from sqlalchemy import and_
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Float, Date, Time
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

from gscap.utils import *


__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'


""" Dark Sky API url """
OPEN_WEATHER_URL = 'http://api.openweathermap.org/data/2.5/weather?'

CONNECTION_RESET_ATTEMPTS = 3
"""How many times to retry a network timeout """

CONNECTION_WAIT_TIME = 3
"""Seconds to wait between each failed network attempt"""

"""Dark Sky hourly call columns"""
HOURLY_COLS = ['day','lon','lat','weather.id','zipcode'
                    ,'weather.main','weather.description','weather.icon','main.temp'
                    ,'main.feels_like','main.pressure','main.humidity','main.temp_min'
                    ,'main.temp_max','main.sea_level','main.grnd_level','wind.speed'
                    ,'wind.deg','wind.gust','clouds.all','rain.1h'
                    ,'rain.3h','snow.1h','snow.3h','sys.type',
                    'sys.id','sys.message','sys.country','sys.sunrise','sys.sunset']
HOURLY_COLS = [r.replace('.','_') for r in HOURLY_COLS]

""" SQLAlchemy declarative base for ORM features """
Base = declarative_base()


class HourlyWeatherReport(Base):
    __tablename__ = 'hourly'

    lon = Column(Float, primary_key=True)
    lat = Column(Float, primary_key=True)
    date = Column(Date, primary_key=True)
    time = Column(Time, primary_key=True)

    base = Column(String)
    zipcode = Column(String)
    weather_id = Column(String)
    weather_main = Column(Float)
    weather_description = Column(String)
    weather_icon = Column(Float)
    main_temp = Column(Float)
    main_feels_like = Column(Float)
    main_pressure = Column(Float)
    main_humidity = Column(Float)
    main_temp_min = Column(Float)
    main_temp_max = Column(Float)
    main_sea_level = Column(Float)
    main_grnd_level = Column(Float)
    wind_speed = Column(Float)
    wind_deg = Column(Float)
    wind_gust = Column(Float)
    clouds_all = Column(String)
    rain_1h = Column(Float)
    rain_3h = Column(Float)
    snow_1h = Column(Float)
    snow_3h = Column(Float)
    sys_type = Column(String)
    sys_id = Column(String)
    sys_message = Column(String)
    sys_country = Column(String)
    sys_sunrise = Column(Float)
    sys_sunset = Column(Float)
    timezone = Column(Float)

    id = Column(Float)

    name = Column(String)
    cod = Column(String)


    @hybrid_property
    def hour(self):
        return self.time.hour

    @property
    def dict(self):
        return dict(
            lon = self.lon,
            lat = self.lat,
            date = self.date,
            time = self.time,
            base = self.base,
            zipcode = self.zipcode,
            weather_id = self.weather_id,
            weather_description = self.weather_description,
            weather_main = self.weather_main,
            main_temp = self.main_temp,
            main_feels_like = self.main_feels_like,
            main_pressure = self.main_pressure,
            main_humidity = self.main_humidity,
            main_temp_min = self.main_temp_min,
            main_temp_max = self.main_temp_max,
            main_sea_level = self.main_sea_level,
            main_grnd_level = self.main_grnd_level,
            wind_speed = self.wind_speed,
            wind_deg = self.wind_deg,
            wind_gust = self.wind_gust,
            clouds_all = self.clouds_all,
            rain_1h = self.rain_1h,
            rain_3h = self.rain_3h,
            snow_1h = self.snow_1h,
            snow_3h = self.snow_3h,
            sys_type = self.sys_type,
            sys_id = self.sys_id,
            sys_country = self.sys_country,
            sys_message = self.sys_message,
            sys_sunrise = self.sys_sunrise,
            sys_sunset = self.sys_sunset,
            id = self.id,
            timezone = self.timezone,
            name = self.name,
            cod = self.cod,
        )

    def from_tuple(self, tup):
        for var in ['base','lon','lat','weather.id','zipcode'
                    ,'weather.main','weather.description','weather.icon','main.temp'
                    ,'main.feels_like','main.pressure','main.humidity','main.temp_min'
                    ,'main.temp_min','main.sea_level','main.grnd_level','wind.speed'
                    ,'wind.deg','wind.gust','clouds.all','rain.1h'
                    ,'rain.3h','snow.1h','snow.3h','sys.type',
                    'sys.id','sys.message','sys.country','sys.sunrise','sys.sunset','timezone','id','name','cod','time','date']:
            if var in list(tup[1].index):
                setattr(self, var.replace('.','_'), tup[1][var])
        return self

    def __repr__(self):
        return f'HWR<(lat={self.lat}, ' \
               f'lon={self.lon}, ' \
               f'date={self.time.isoformat()})>'


@contextmanager
def session_scope(engine_):
    """Provide a transactional scope around a series of operations."""
    if engine_ == WCNAME:
        engine_ = create_engine(WCNAME, module=sqlite)
    else:
        engine_ = create_engine(engine_, module=sqlite)

    Session = sessionmaker(bind=engine_)
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


WCNAME = f'sqlite+pysqlite:///{dpath("weather_cache_new.sqlite")}'
Base.metadata.create_all(
    create_engine(WCNAME, module=sqlite)
)


"""All weather requests should come as the following namedtuple"""
WeatherRequest = namedtuple(
    'WeatherRequest', ['date', 'lat', 'lon', 'zipcode']
)
WEATHER_REQUEST_ERROR_MESSAGE = \
    'Request must be a 2 or 3-tuple containing two floats to represent the latitude, ' +\
    'longitude coordinate or an integer to represent the zipcode and a DateTime object ' +\
    'for the date to request'


def cache_manager(request_qu, response_qu, engine_):
    finished = False

    while not finished:
        with session_scope(engine_) as session:
            try:
                r = request_qu.get()
            except EOFError:
                finished = True

            if r['type'] == 'get':
                date, lat, lon = r['args']
                content = get_from_cache(date, lat, lon, session)
                response_qu.put(dict(
                        pid=r['pid'],
                        response=content
                ))

            elif r['type'] == 'put':
                put_to_cache(r['args'][0], session)

            elif r['type'] == 'end':
                finished = True


def weather_report(
        request, summarize='daily', n_jobs=1, progress_qu=None, kwargs=None
):

    """retrieve a weather report for specific time and location

    Args:
        request: (int, DateTime) or (float, float, DateTime) or list containing 2 or 3 tuples with each element indicating a unique request. Each 2-tuple argument will be treated as (zipcode, DateTime) while each 3-tuple argument will be treated as (latitude, longitude, DateTime). Latitude and longitudes must be supplied in degree decimal format.
        summarize: (str) {'none', 'daily''}
        n_jobs: (int) number of procs to use (-1 for all)
        progress_qu: (multiprocessing.Queue)
        kwargs: dict used for development and testing
    Raises:
        ValueError if lat/lon is outside of valid range
        ValueError if frequency is not in { None, 'daily' }

    Returns:
        pd.DataFrame containing weather report
    """
    # validate the request
    request = verify_request(request)

    man = mul.Manager()
    request_qu, response_qu = man.Queue(), man.Queue()

    if kwargs is not None and 'engine' in kwargs.keys():
        engine_ = kwargs['engine']
    else:
        engine_ = WCNAME

    cman = mul.Process(
        target=cache_manager, args=(request_qu, response_qu, engine_)
    )
    cman.start()

    cpus = mul.cpu_count()
    cpus = cpus if n_jobs == -1 or n_jobs >= cpus else n_jobs
    pool = mul.Pool(cpus)

    # maintain the current behavior if only one day/lat/lon is requested
    if len(request) == 1:
        results = [process_request(
            (request[0], progress_qu, request_qu, response_qu)
        )]
        request_qu.put(dict(type='end'))

    # otherwise startup some parallel tasks
    else:
        results = list(pool.map(
            process_request,
            [(ri, progress_qu, request_qu, response_qu) for ri in request]
        ))
        request_qu.put(dict(type='end'))

    if summarize in ['daily']:
        hits = np.sum([r.get('hits') for r in results])
        misses = np.sum([r.get('misses') for r in results])

        r_ = list(pool.map(
            summarize_report, [(c, ri) for c, ri in zip(results, request)]
        ))
        report = pd.DataFrame([(r.get('report')) for r in r_])

        r_cols = set(report.columns) - {'date', 'lat', 'lon', 'zipcode', 'summary'}

        # safely round the results
        for c in r_cols:
            try:
                report[c] = [np.round(vi, 2) for vi in report[c]]
            except:
                pass

        results = dict(report=report, hits=hits, misses=misses)
    else:
        results = results[0]

    pool.terminate(); pool.join(); del pool
    cman.terminate(); cman.join(); del cman
    man.shutdown(); del man

    return results[0] if len(results) == 1 else results


def get_from_cache(date, lat, lon, session):
    day = date.date()

    content = session.query(HourlyWeatherReport).filter(and_(
        (HourlyWeatherReport.lat == lat),
        (HourlyWeatherReport.lon == lon),
        (HourlyWeatherReport.date == day)
    )).all()

    if len(content) > 0:
        content = pd.DataFrame([ri.dict for ri in content])
        return content

    else:
        return None


def make_empty(day):
    c = pd.DataFrame(columns=HOURLY_COLS)
    t = {k: np.nan for k in HOURLY_COLS}
    c.loc[-1] = t

    c['lat'], c['lon'], c['day'] = 0, 0, day

    # ensure any columns not returned from dark sky are still in the df
    missing_cols = list(set(HOURLY_COLS) - set(c.columns))
    for ci in missing_cols:
        c[ci] = np.nan

    return c


def summarize_report(args):
    c, ri = args
    r = c.get('report').to_dict(orient='records')[0]
    zip = r['zipcode']
    for ke in ['weather','lat','lon','date','zipcode']:
        r.pop(ke, None)

    dm = dict(
        date=ri.date,
        lat=ri.lat,
        lon=ri.lon,
        zipcode = zip,
        **r
    )
    return dict(report=dm, hits=c.get('hits'), misses=c.get('misses'))


def process_request(args):
    request, progress_qu, request_qu, response_qu = args
    CONFIG = load_config_file()

    key = CONFIG['OpenWeatherMapAPI']

    if request is None:
        update_progress(progress_qu)
        return None

    day = dt.datetime(
        year=request.date.year,
        month=request.date.month,
        day=request.date.day,
        hour=12
    )

    my_pid = os.getpid()
    request_qu.put(dict(pid=my_pid, type='get', args=(
        day, request.lat, request.lon
    )))
    r = response_qu.get()

    while r['pid'] != my_pid:
        response_qu.put(r)
        r = response_qu.get()

    if r['response'] is not None:
        update_progress(progress_qu)
        r['response']['zipcode'] = request.zipcode
        return dict(report=r['response'], hits=1, misses=0)
    else:
        pass

    # if we made it this far, we had a cache miss so make the request
    a, call_complete = 0, False
    while not call_complete:
        try:
            url = f'{OPEN_WEATHER_URL}lat={request.lat}&lon={request.lon}&dt={int(day.timestamp())}&appid={key}'
            r = requests.get(url)

            call_complete = True
        except ConnectionError:
            a += 1
            time.sleep(CONNECTION_WAIT_TIME)

            if a == CONNECTION_RESET_ATTEMPTS:
                raise ConnectionError
        except Exception as e:
            raise e

    # make sure the web result is valid
    if r.ok:
        j = dict(r.json())

        if j is not None:
            c = pd.json_normalize(j, sep='_')
    else:
        c = pd.DataFrame(columns=HOURLY_COLS)
        t = {k: np.nan for k in HOURLY_COLS}
        c['time'] = dt.time(hour=12, minute=0, second=0)
        c.loc[-1] = t

    # cache the results
    c['lat'], c['lon'], c['date'] = request.lat, request.lon, day.date()
    c['zipcode'] = request.zipcode

    request_qu.put(dict(type='put', args=(c,)))

    update_progress(progress_qu)
    return dict(report=c, hits=0, misses=1)


def put_to_cache(content, session):
    rows = [
        HourlyWeatherReport().from_tuple(t)
        for t in content.iterrows()
    ]
    session.add_all(rows)


def verify_request(request):
    if not isinstance(request, list):
        request = [request]

    def check_one(r):
        if len(r) == 2:
            return verify_zipcode_date_request(r)
        elif len(r) == 3:
            return verify_latlon_date_request(r)
        else:
            raise ValueError('Only tuples of size 2 or 3 are permitted')

    request = list(map(check_one, request))
    return request


def verify_zipcode_date_request(request):
    if len(request) != 2:
        raise ValueError(WEATHER_REQUEST_ERROR_MESSAGE)

    zc, d = None, None
    for i in request:
        if isinstance(i, int) or isint(i):
            zc = int(i)
        elif isinstance(i, dt.datetime):
            d = i

    if d is None or zc is None:
        raise ValueError(WEATHER_REQUEST_ERROR_MESSAGE)

    lat, lon = dd_from_zip(zc)

    if lat == lon == 0:
        print(f'zip <{zc}> was not found in db. skipping this request')
        return None

    return WeatherRequest(lat=lat, lon=lon, zipcode=zc, date=d)


def verify_latlon_date_request(request):
    if len(request) != 3:
        raise ValueError(WEATHER_REQUEST_ERROR_MESSAGE)

    lat, lon, d = None, None, None
    for i in request:
        if isinstance(i, (int, float)) or isfloat(i):
            if lat is None:
                lat = float(i)
            elif lon is None:
                lon = float(i)
        elif isinstance(i, dt.datetime):
            d = i

    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise ValueError('lat, lon must be in a valid range')

    if lat is None or lon is None or d is None:
        raise ValueError(WEATHER_REQUEST_ERROR_MESSAGE)

    zc = zip_from_dd(lat, lon, suppress_warnings=True)
    return WeatherRequest(lat=lat, lon=lon, zipcode=zc, date=d)


def update_progress(progress_queue):
    if progress_queue is not None:
        progress_queue.put(1)
    else:
        pass


if __name__ == '__main__':
    today = dt.datetime.now()
    requests = [
        (31.11, -82.92, today - dt.timedelta(days=2)),
        (today - dt.timedelta(days=3), 33.11, -82.92),
        (31.11, today - dt.timedelta(days=4), -81.92),
    ]
    request = (47579, today - dt.timedelta(days=1))
    report = weather_report(requests)
    print(report)

