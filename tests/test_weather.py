import datetime as dt
import multiprocessing as mul
import os
from sqlite3 import dbapi2 as sqlite
from unittest import skip, TestCase

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError
import responses
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

import gscapl.weather as wthr


class TestWeather(TestCase):
    """test class for
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization

        """
        now = dt.datetime(
            year=2005, month=5, day=5, hour=12
        )
        cls.day = dt.datetime(year=now.year, month=now.month, day=now.day)
        cls.time = now.time()
        cls.lat = 32.3788
        cls.lon = -84.90685
        cls.zipcode = 31905

        cls.wrows = pd.DataFrame(
            {k: np.nan for k in wthr.HOURLY_COLS},
            index=[0]
        )
        cls.wrows['lat'] = cls.lat
        cls.wrows['lon'] = cls.lon
        cls.wrows['time'] = dt.datetime(year=2005, month=6, day=5)

        cls.wcname = 'sqlite+pysqlite:///test_weather_cache.sqlite'
        cls.del_cache()

        engine = create_engine(cls.wcname, module=sqlite)
        wthr.Base.metadata.create_all(engine)
        cls.session = sessionmaker(bind=engine)()

        cls.cache_kwargs = {'engine': cls.wcname}

        wthr.CONNECTION_RESET_ATTEMPTS = 1
        wthr.CONNECTION_WAIT_TIME = 0

    @classmethod
    def del_cache(cls):
        fn = cls.wcname.replace('sqlite+pysqlite:///', '')
        if os.path.exists(fn):
            os.remove(fn)

    @classmethod
    def mock_darksy(cls):
        fn = 'mock_darksky_response'
        if not os.path.exists(fn):
            fn = os.path.join('tests', fn)

        with open(fn, 'r') as f:
            mock_response = f.read()

        return mock_response

    @classmethod
    def tearDownClass(cls):
        """perform when all tests are complete
        """
        del cls.session
        cls.del_cache()

    def setUp(self):
        """perform before each unittest"""
        pass

    def tearDown(self):
        """perform after each unittest
        """
        t = self.session.query(wthr.HourlyWeatherReport).all()
        for ti in t:
            self.session.delete(ti)
        self.session.commit()

    def test_verify_zipcode_date_request(self):
        request = (self.zipcode, self.day)
        request = wthr.verify_zipcode_date_request(request)
        self.assertTrue(isinstance(request, wthr.WeatherRequest))

        request = (self.day, self.zipcode)
        request = wthr.verify_zipcode_date_request(request)
        self.assertTrue(isinstance(request, wthr.WeatherRequest))

        request = (self.day, '47579')
        request = wthr.verify_zipcode_date_request(request)
        self.assertTrue(isinstance(request, wthr.WeatherRequest))

    def test_verify_zipcode_date_request_throws_1(self):
        request = (self.lat, self.lon, self.day)
        self.assertRaises(ValueError, wthr.verify_zipcode_date_request, (request,))

    def test_verify_zipcode_date_request_throws_2(self):
        request = (self.lat, self.lon)
        self.assertRaises(ValueError, wthr.verify_zipcode_date_request, (request,))

    def test_verify_location(self):
        request = (self.zipcode, self.day)
        request = wthr.verify_request(request)
        self.assertTrue(isinstance(request, list))
        self.assertTrue(all(isinstance(i, wthr.WeatherRequest) for i in request))

    def test_verify_location_many(self):
        request = [
            (self.zipcode, self.day),
            (self.day, self.zipcode),
            (self.lat, self.lon, self.day),
            (self.day, self.lat, self.lon),
            (self.lat, self.day, self.lon)
        ]
        request = wthr.verify_request(request)
        self.assertTrue(isinstance(request, list))
        self.assertTrue(all(isinstance(i, wthr.WeatherRequest) for i in request))
        self.assertTrue(all(
            i.lat == self.lat and i.lon == self.lon and i.zipcode == self.zipcode
            for i in request
        ))

    @responses.activate
    def test_hourly_weather_report_ll(self):
        hwr = wthr.HourlyWeatherReport(
            lat=self.lat, lon=self.lon, date=self.day, time=self.time
        )
        self.session.add(hwr)
        self.session.commit()

        report = wthr.weather_report((self.lat, self.lon, self.day), kwargs=self.cache_kwargs)

        self.assertTrue(report is not None)
        self.assertTrue(isinstance(report, dict))
        self.assertTrue(
            all(k in ['report', 'hits', 'misses'] for k in report.keys())
        )
        self.assertTrue(isinstance(report.get('report'), pd.DataFrame))
        self.assertTrue(report.get('report').shape is not (0, 0))

    @responses.activate
    def test_hourly_weather_report_no_summary(self):
        hwr = wthr.HourlyWeatherReport(
            lat=self.lat, lon=self.lon, date=self.day, time=self.time
        )
        self.session.add(hwr)
        self.session.commit()

        report = wthr.weather_report((self.lat, self.lon, self.day), summarize='none', kwargs=self.cache_kwargs)
        self.assertTrue(report is not None)
        self.assertTrue(isinstance(report, dict))
        self.assertTrue(
            all(k in ['report', 'hits', 'misses'] for k in report.keys())
        )
        self.assertTrue(isinstance(report.get('report'), pd.DataFrame))
        self.assertTrue(report.get('report').shape is not (0, 0))

    def test_update_progress(self):
        qu = mul.Queue()
        t = wthr.update_progress(qu)
        t = qu.get(timeout=1)
        self.assertTrue(t == 1)

    def test_session_scope(self):
        hwr = wthr.HourlyWeatherReport(
            lat=self.lat,
            lon=self.lon,
            date=self.day,
            time=self.day.time()
        )

        with wthr.session_scope(self.wcname) as session:
            self.assertTrue(os.path.exists(self.wcname.replace('sqlite+pysqlite:///', '')))
            self.assertTrue(isinstance(session, Session))
            session.add(hwr)

        t = self.session.query(wthr.HourlyWeatherReport).all()
        self.assertTrue(len(t) == 1)

        with wthr.session_scope(self.wcname) as session:
            session.add(hwr)
            session.add(hwr)

        t = self.session.query(wthr.HourlyWeatherReport).all()
        self.assertTrue(len(t) == 1)

    def test_exchange_with_cache(self):
        content = pd.DataFrame([
            wthr.HourlyWeatherReport(
                lat=self.lat,
                lon=self.lon,
                date=(self.day-dt.timedelta(days=i)).date(),
                time=self.time
            ).dict
            for i in range(2)
        ])
        wthr.put_to_cache(content, self.session)

        t = self.session.query(wthr.HourlyWeatherReport).all()
        self.assertTrue(len(t) == 2)

        for i in range(2):
            t = wthr.get_from_cache(
                date=self.day-dt.timedelta(days=i),
                lat=self.lat,
                lon=self.lon,
                session=self.session
            )
            self.assertTrue(isinstance(t, pd.DataFrame))
            self.assertTrue(len(t) == 1)

    def test_cache_man(self):
        reqqu = mul.Queue()
        resqu = mul.Queue()

        content = pd.DataFrame([
            wthr.HourlyWeatherReport(
                lat=self.lat,
                lon=self.lon,
                date=self.day,
                time=self.time
            ).dict
        ])

        req = dict(
            type='put',
            args=[content]
        )
        reqqu.put(req)

        req = dict(
            pid=os.getpid(),
            type='get',
            args=(
                self.day,
                self.lat,
                self.lon
            )
        )
        reqqu.put(req)
        reqqu.put(dict(type='end'))

        wthr.cache_manager(reqqu, resqu, self.wcname)

        res = resqu.get()
        self.assertTrue(res['pid'] == os.getpid())

    def test_make_empty(self):
        t = wthr.make_empty(self.day)

        self.assertTrue(isinstance(t, pd.DataFrame))
        self.assertTrue(len(t.columns) == len(wthr.HOURLY_COLS))
        self.assertTrue(len(t) == 1)

    def test_summarize_report(self):
        report = pd.DataFrame([
            wthr.HourlyWeatherReport(
                lat=self.lat, lon=self.lon,
                date=self.day - dt.timedelta(hours=i),
                time=self.time
            ).dict
            for i in range(23)
        ])

        cols = list(set(wthr.HOURLY_COLS)-{'lat', 'lon', 'date', 'time'})
        for c in cols:
            report[c] = .5

        req = wthr.WeatherRequest(
            lat=self.lat, lon=self.lon, date=self.day, zipcode=None
        )

        args = (
            dict(report=report),
            req
        )

        t = wthr.summarize_report(args)

        cols = list({k for k in t['report'].keys()}-{
            'lat', 'lon', 'date', 'zipcode', 'precip_sum'
        })

        for c in cols:
            if 'IQR' in c or 'std' in c:
                self.assertTrue(t['report'][c] == 0.0)
            else:
                self.assertTrue(t['report'][c] == 0.5)
        self.assertTrue(t['report']['precip_sum'] == 11.5)

    def test_verify_request(self):
        r = wthr.verify_request((0, 0, self.day))
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], wthr.WeatherRequest))

        r = wthr.verify_request((self.zipcode, self.day))
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], wthr.WeatherRequest))

        self.assertRaises(ValueError, wthr.verify_request, (9,))

    def test_process_request_from_cache(self):
        hwr = wthr.HourlyWeatherReport(
            lat=self.lat, lon=self.lon, date=self.day, time=self.time
        )
        self.session.add(hwr)
        self.session.commit()

        proqu, reqque, resque = mul.Queue(), mul.Queue(), mul.Queue()

        # check None reponse
        args = (None, proqu, reqque, resque)
        self.assertTrue(wthr.process_request(args) is None)

        # check cached response
        req = wthr.WeatherRequest(
            lat=self.lat, lon=self.lon, date=self.day, zipcode=None
        )
        args = (req, proqu, reqque, resque)
        resque.put(dict(
            pid=os.getpid(),
            response=dict(zipcode=self.zipcode)
        ))
        t = wthr.process_request(args)
        self.assertTrue(all(
            c in t.keys() for c in ['report', 'hits', 'misses']
        ))
        self.assertTrue(t['hits'] == 1 and t['misses'] == 0)

    @responses.activate
    def test_process_request(self):
        key = wthr.CONFIG['DarkSkyAPI']

        for url in [
            f'https://api.darksky.net/forecast/{key}/32.4,-84.9,1115319600',
            f'https://api.darksky.net/forecast/{key}/32.4,-84.9,1115294400'
        ]:
            responses.add(
                responses.GET,
                url,
                body=self.mock_darksy(),
                status=200
            )

        lat = np.round(self.lat, 1)
        lon = np.round(self.lon, 1)

        proqu, reqque, resque = mul.Queue(), mul.Queue(), mul.Queue()
        req = wthr.WeatherRequest(
            lat=lat, lon=lon, date=self.day, zipcode=None
        )

        args = (req, proqu, reqque, resque)
        resque.put(dict(
            pid=os.getpid(),
            response=None
        ))
        t = wthr.process_request(args)

        # check the direct response
        self.assertTrue(all(
            c in t.keys() for c in ['report', 'hits', 'misses']
        ))
        self.assertTrue(t['misses'] == 1 and t['hits'] == 0)
        self.assertTrue(isinstance(t['report'], pd.DataFrame))
        self.assertTrue(len(t['report']) == 24)

        def empty_and_close(qu):
            while not qu.empty():
                qu.get()
            qu.close()

        empty_and_close(proqu)
        empty_and_close(reqque)
        empty_and_close(resque)

        del proqu, reqque, resque

    @responses.activate
    def test_process_request_raises_connection_error(self):
        key = wthr.CONFIG['DarkSkyAPI']
        lat = np.round(self.lat, 1)
        lon = np.round(self.lon, 1)
        url = f'https://api.darksky.net/forecast/{key}/32.4,-84.9,1115319600'

        responses.add(
            responses.GET,
            url,
            body=ConnectionError('...')
        )

        proqu, reqque, resque = mul.Queue(), mul.Queue(), mul.Queue()
        req = wthr.WeatherRequest(
            lat=lat, lon=lon, date=self.day, zipcode=None
        )

        args = (req, proqu, reqque, resque)
        resque.put(dict(
            pid=os.getpid(),
            response=None
        ))
        self.assertRaises(ConnectionError, wthr.process_request, args)

    @responses.activate
    def test_process_request_darksky_failed(self):
        key = wthr.CONFIG['DarkSkyAPI']
        lat = np.round(self.lat, 1)
        lon = np.round(self.lon, 1)
        url = f'https://api.darksky.net/forecast/{key}/32.4,-84.9,1115319600'

        responses.add(
            responses.GET,
            url,
            status=404
        )

        proqu, reqque, resque = mul.Queue(), mul.Queue(), mul.Queue()
        req = wthr.WeatherRequest(
            lat=lat, lon=lon, date=self.day, zipcode=None
        )

        args = (req, proqu, reqque, resque)
        resque.put(dict(
            pid=os.getpid(),
            response=None
        ))
        t = wthr.process_request(args)
        self.assertTrue(all(
            c in t.keys() for c in ['report', 'hits', 'misses']
        ))
        self.assertTrue(isinstance(t['report'], pd.DataFrame))
        for c in t['report'].columns:
            if c in ['date', 'lat', 'lon']:
                self.assertTrue(all([not b for b in pd.isnull(t['report'][c])]))
            else:
                self.assertTrue(all(pd.isnull(t['report'][c])))