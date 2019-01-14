#!/usr/bin/env python

""" A collection of scripts for processing weather data """

from collections import namedtuple
import multiprocessing as mul
import requests
from sqlite3 import dbapi2 as sqlite
import time
from urllib3.exceptions import NewConnectionError

from sqlalchemy import and_
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

from gscapl.utils import *


__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'


""" Dark Sky API url """
DARK_SKY_URL = 'https://api.darksky.net/forecast'

"""How many times to retry a network timeout 
and how many seconds to wait between each """
CONNECTION_RESET_ATTEMPTS = 99
CONNECTION_WAIT_TIME = 60


"""Dark Sky hourly call columns"""
HOURLY_COLS = [
    'time', 'summary', 'icon', 'precipIntensity', 'precipProbability',
    'temperature', 'apparentTemperature', 'dewPoint', 'humidity',
    'pressure', 'pressureError', 'windSpeed', 'windBearing', 'cloudCover',
    'cloudCoverError', 'uvIndex', 'visibility', 'lat', 'lon', 'day', 'hour',
    'ozone', 'precipAccumulation', 'precipType', 'temperatureError',
    'windBearingError', 'windGust', 'windSpeedError'

]

""" SQLAlchemy declarative base for ORM features """
Base = declarative_base()


class HourlyWeatherReport(Base):
    __tablename__ = 'hourly'

    lat = Column(Float, primary_key=True)
    lon = Column(Float, primary_key=True)
    time = Column(DateTime, primary_key=True)

    apparentTemperature = Column(Float)
    cloudCover = Column(Float)
    cloudCoverError = Column(Float)
    dewPoint = Column(Float)
    humidity = Column(Float)
    icon = Column(String)
    ozone = Column(Float)
    precipAccumulation = Column(Float)
    precipIntensity = Column(Float)
    precipProbability = Column(Float)
    precipType = Column(String)
    pressure = Column(Float)
    pressureError = Column(Float)
    summary = Column(String)
    temperature = Column(Float)
    temperatureError = Column(Float)
    uvIndex = Column(Float)
    visibility = Column(Float)
    windBearing = Column(Float)
    windBearingError = Column(Float)
    windGust = Column(Float)
    windSpeed = Column(Float)
    windSpeedError = Column(Float)

    @hybrid_property
    def day(self):
        return self.time.date()

    @hybrid_property
    def hour(self):
        return self.time.hour

    @property
    def dict(self):
        return dict(
            lat=self.lat,
            lon=self.lon,
            time=self.time,
            day=self.day,
            apparentTemperature=self.apparentTemperature,
            cloudCover=self.cloudCover,
            cloudCoverError=self.cloudCoverError,
            dewPoint=self.dewPoint,
            hour=self.hour,
            humidity=self.humidity,
            icon=self.icon,
            ozone=self.ozone,
            precipAccumulation=self.precipAccumulation,
            precipIntensity=self.precipIntensity,
            precipProbability=self.precipProbability,
            precipType=self.precipType,
            pressure=self.pressure,
            pressureError=self.pressureError,
            summary=self.summary,
            temperature=self.temperature,
            temperatureError=self.temperatureError,
            uvIndex=self.uvIndex,
            visibility=self.visibility,
            windBearing=self.windBearing,
            windBearingError=self.windBearingError,
            windGust=self.windGust,
            windSpeed=self.windSpeed,
            windSpeedError=self.windSpeedError
        )

    def from_tuple(self, tup):
        self.lat = tup.lat
        self.lon = tup.lon
        self.time = tup.day
        self.apparentTemperature = tup.apparentTemperature
        self.cloudCover = tup.cloudCover
        self.cloudCoverError = tup.cloudCoverError
        self.dewPoint = tup.dewPoint
        self.humidity = tup.humidity
        self.icon = tup.icon
        self.ozone = tup.ozone
        self.precipAccumulation = tup.precipAccumulation
        self.precipIntensity = tup.precipIntensity
        self.precipProbability = tup.precipProbability
        self.precipType = tup.precipType
        self.pressure = tup.pressure
        self.pressureError = tup.pressureError
        self.summary = tup.summary
        self.temperature = tup.temperature
        self.temperatureError = tup.temperatureError
        self.uvIndex = tup.uvIndex
        self.visibility = tup.visibility
        self.windBearing = tup.windBearing
        self.windBearingError = tup.windBearingError
        self.windGust = tup.windGust
        self.windSpeed = tup.windSpeed
        self.windSpeedError = tup.windSpeedError

        return self

    def __repr__(self):
        return f'HWR<(lat={self.lat}, ' \
               f'lon={self.lon}, ' \
               f'date={self.time.isoformat()})>'


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


wcname = 'weather_cache.sqlite'
engine = create_engine(f'sqlite+pysqlite:///{dpath(wcname)}', module=sqlite)
Base.metadata.create_all(engine)
del wcname


"""All weather requests should come as the following namedtuple"""
WeatherRequest = namedtuple(
    'WeatherRequest', ['date', 'lat', 'lon', 'zipcode']
)


def cache_manager(request_qu, response_qu):
    finished = False

    while not finished:
        with session_scope() as session:
            try:
                r = request_qu.get()
            except EOFError:
                finished = True

            if r['type'] == 'get':
                date, lat, lon = r['args']
                content = get_from_cache(date, lat, lon, session)
                response_qu.put(
                    dict(
                        pid=r['pid'],
                        response=content)
                )

            elif r['type'] == 'put':
                put_to_cache(r['args'][0], session)

            elif r['type'] == 'end':
                finished = True


def weather_report(
        request, summarize='daily', n_jobs=1, progress_qu=None
):

    """retrieve a weather report for specific lat, lon, and day

    Args:
        request: (weather.WeatherRequest)
        summarize: (str) {'daily', 'weekly'}
        n_jobs: (int) number of procs to use (-1 for all)
        key: (str)
        progress_qu: (multiprocessing.Queue)

    Raises:
        ValueError if lat/lon is outside of valid range
        ValueError if frequency is not in { 'hourly', 'daily' }

    Returns:
        pd.DataFrame containing hourly (24 rows) or daily (1 row) data else None
    """
    # make sure the request is in the correct data type
    if not isinstance(request, list):
        request = [request]
    else:
        pass

    man = mul.Manager()
    request_qu, response_qu = man.Queue(), man.Queue()
    cman = mul.Process(
        target=cache_manager, args=(request_qu, response_qu)
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

    if summarize in ['daily', 'weekly']:
        hits = np.sum([r.get('hits') for r in results])
        misses = np.sum([r.get('misses') for r in results])

        if summarize == 'weekly':
            results = pd.concat([r.get('report') for r in results], sort=True)
            results = results.sort_values(by='time')

            # break into weekly results
            results['week'] = [d.week for d in results.time]
            results = [
                dict(
                    report=pd.DataFrame(g).drop(columns='week'),
                    hits=0,
                    misses=0
                )
                for n, g in results.groupby(['lat', 'lon', 'week'])
            ]
            request = [
                WeatherRequest(
                    date=np.min(d.time),
                    lat=d.lat.values[0],
                    lon=d.lon.values[0],
                    zipcode=d.zipcode.values[0]
                )
                for d in [r.get('report') for r in results]
            ]
        else:
            pass

        r_ = list(pool.map(
            summarize_report, [(c, ri) for c, ri in zip(results, request)]
        ))
        report = pd.DataFrame([(r.get('report')) for r in r_])

        r_cols = set(report.columns) - {'date', 'lat', 'lon', 'zipcode', 'summary'}

        # safely round the results
        for c in r_cols:
            report[c] = [np.round(vi, 2) if isfloat(vi) else np.nan for vi in report[c]]

        results = dict(report=report, hits=hits, misses=misses)

    pool.terminate(); pool.join(); del pool
    cman.terminate(); cman.join(); del cman
    man.shutdown(); del man

    return results[0] if len(results) == 1 else results


def get_from_cache(date, lat, lon, session):
    day = date.date()

    content = session.query(HourlyWeatherReport).filter(and_(
        (HourlyWeatherReport.lat == lat),
        (HourlyWeatherReport.lon == lon),
        #(cast(HourlyWeatherReport.time, Date) == cast(day, Date))
        #(HourlyWeatherReport.time == func.datetime(day))
    )).all()

    # this is incredibly inefficient but both methods above fail to return results for all queries
    content = [c for c in content if c.time.date() == day]

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
    """

    Args:
        args:

    Returns:

    """
    c, ri = args
    r = c.get('report')

    def vstats(vals):
        vals = [v if isfloat(v) else np.nan for v in vals]
        valid = any([not pd.isnull(vi) for vi in vals])

        try:
            qs  = np.nanpercentile(vals, [25, 50, 75], interpolation='nearest') \
                if valid else [np.nan, np.nan, np.nan]
            iqr = qs[2] - qs[0]
            med = qs[1]
            m   = np.nanmean(vals) if valid else np.nan
            s   = np.nanstd(vals)  if valid else np.nan
        except TypeError:
            med, iqr, m, s = 'err', 'err', 'err', 'err'

        return med, iqr, m, s

    cc_med, cc_iqr, cc_mean, cc_std = vstats(r.cloudCover)
    dp_med, dp_iqr, dp_mean, dp_std = vstats(r.dewPoint)
    h_med, h_iqr, h_mean, h_std = vstats(r.humidity)
    t_med, t_iqr, t_mean, t_std = vstats(r.temperature)
    p_sum  = np.sum([v  for v in r.precipIntensity if isfloat(v)])

    # summary = ' '.join(list(set([str(v) for v in r.icon.values])))
    # summary = ', '.join(
    #     list(set(re.sub(r'(,)+', ' ', summary).split(' '))))

    lat, lon, zipcode = verify_location(ri)

    dm = dict(
        date=ri.date,
        lat=lat,
        lon=lon,
        zipcode=zipcode,
        cloud_cover_mean=cc_mean,
        cloud_cover_std=cc_std,
        cloud_cover_median=cc_med,
        cloud_cover_IQR=cc_iqr,
        dew_point_mean=dp_mean,
        dew_point_std=dp_std,
        dew_point_median=dp_med,
        dew_point_IQR=dp_iqr,
        humidity_mean=h_mean,
        humidity_std=h_std,
        humidity_median=h_med,
        humidity_IQR=h_iqr,
        precip_sum=p_sum,
        temp_mean=t_mean,
        temp_std=t_std,
        temp_med=t_med,
        temp_IQR=t_iqr
    )
    return dict(report=dm, hits=c.get('hits'), misses=c.get('misses'))


def process_request(args):
    request, progress_qu, request_qu, response_qu = args

    key = CONFIG['DarkSkyAPI']

    d = request.date
    day = dt.datetime(
        year=d.year, month=d.month, day=d.day, hour=12
    )

    lat, lon, zipcode = verify_location(request)
    if lat == lon == 0:
        return make_empty(day)

    my_pid = os.getpid()
    request_qu.put(dict(pid=my_pid, type='get', args=(day, lat, lon)))
    r = response_qu.get()

    while r['pid'] != my_pid:
        response_qu.put(r)
        r = response_qu.get()

    if r['response'] is not None:
        if progress_qu is not None:
            progress_qu.put(1)
        else:
            pass

        r['response']['zipcode'] = zipcode
        return dict(report=r['response'], hits=1, misses=0)
    else:
        pass

    # if we made it this far, we had a cache miss so make the request
    # wrapping it to catch and retry the request if the connection errors out
    # a fooled man don't get fooled again
    a, call_complete = 0, False
    while a < CONNECTION_RESET_ATTEMPTS and not call_complete:
        try:
            r = requests.get(
                f'{DARK_SKY_URL}/{key}/{lat},{lon},{int(day.timestamp())}'
            )
            call_complete = True
        except NewConnectionError:
            a += 1
            time.sleep(CONNECTION_WAIT_TIME)
        except Exception as e:
            print(f'\n{a} connection reset attempts failed\n')
            raise e

    # make sure the web result is valid
    if r.ok:
        j = dict(r.json())

        if j is not None and j.get('hourly') is not None:
            c = pd.DataFrame(
                j.get('hourly').get('data')
            )
            c.time = c.time.apply(
                lambda r: dt.datetime.fromtimestamp(r)
            )
            c['hour'] = c.time.apply(lambda r: r.hour)
        else:
            c = pd.DataFrame(columns=HOURLY_COLS)
            t = {k: np.nan for k in HOURLY_COLS}
            c.loc[-1] = t
    else:
        c = pd.DataFrame(columns=HOURLY_COLS)
        t = {k: np.nan for k in HOURLY_COLS}
        c.loc[-1] = t

    # cache our results
    if pd.isnull(day):
        raise Exception(str(request))
    c['lat'], c['lon'], c['day'] = lat, lon, day

    # ensure any columns not returned from dark sky are still in the df
    missing_cols = list(set(HOURLY_COLS) - set(c.columns))
    for ci in missing_cols:
        c[ci] = np.nan

    if any(c.day.isnull()):
        raise Exception(str(request))
    request_qu.put(dict(type='put', args=(c,)))

    # if monitoring using the progress_bar in utils and a qu is supplied
    if progress_qu is not None:
        progress_qu.put(1)
    else:
        pass

    c['zipcode'] = zipcode
    return dict(report=c, hits=0, misses=1)


def put_to_cache(content, session):
    rows = [
        HourlyWeatherReport().from_tuple(t)
        for t in content.itertuples()
    ]
    session.add_all(rows)


def verify_location(request):
    try:
        lat, lon, zipcode = request.lat, request.lon, request.zipcode
    except AttributeError:
        raise ValueError('either lat, lon or zip must be supplied')

    if (lat is None or lon is None) and zipcode is None:
        raise ValueError('either lat, lon or zip must be supplied')

    if (lat is not None and lon is not None) and \
            (lat < -90 or lat > 90 or lon < -180 or lon > 180):
        raise ValueError('lat, lon must be in a valid range')

    if zipcode is not None:
        lat, lon = dd_from_zip(zipcode)

        if lat == lon == 0:
            print(f'zip <{zipcode}> was not found in db')
    else:
        pass

    lat = np.round(lat, 1)
    lon = np.round(lon, 1)

    return lat, lon, zipcode


def remove_null_island():
    with session_scope() as session:
        session.query(HourlyWeatherReport).filter(and_(
            (HourlyWeatherReport.lat == 0),
            (HourlyWeatherReport.lon == 0)
        )).delete()


if __name__ == '__main__':
    pass
