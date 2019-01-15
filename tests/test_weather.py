import datetime as dt
import os
from sqlite3 import dbapi2 as sqlite
from unittest import skip, TestCase

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import gscapl.weather as wthr


class TestWeather(TestCase):
    """test class for
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization

        """
        now = dt.datetime.now()
        cls.day = dt.datetime(year=now.year, month=now.month, day=now.day)
        cls.lat = 32.3788
        cls.lon = -84.90685
        cls.zipcode = 31905

        cls.wrequest_ll = (cls.day, cls.lat, cls.lon)
        cls.wrequest_zc = (cls.day, 98115)

        cls.wrows = pd.DataFrame(
            {k: np.nan for k in wthr.HOURLY_COLS},
            index=[0]
        )
        cls.wrows['lat'] = cls.lat
        cls.wrows['lon'] = cls.lon
        cls.wrows['time'] = dt.datetime(year=2005, month=6, day=5)

        cls.wcname = 'sqlite+pysqlite:///test_weather_cache.sqlite'
        cls.engine = create_engine(cls.wcname, module=sqlite)
        wthr.Base.metadata.create_all(cls.engine)

        cls.cache_kwargs = {'engine': cls.wcname}

    @classmethod
    def tearDownClass(cls):
        """perform when all tests are complete
        """
        pass
        # if os.path.exists(cls.wcname):
        #     os.remove(cls.wcname)

    def setUp(self):
        """perform before each unittest"""
        pass

    def tearDown(self):
        """perform after each unittest
        """
        pass

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

    @skip
    def test_hourly_weather_report_ll(self):
        report = wthr.weather_report(self.wrequest_ll, kwargs=self.cache_kwargs)
        self.assertTrue(report is not None)
        self.assertTrue(isinstance(report, dict))
        self.assertTrue(
            all(k in ['report', 'hits', 'misses'] for k in report.keys())
        )
        self.assertTrue(isinstance(report.get('report'), pd.DataFrame))
        self.assertTrue(report.get('report').shape is not (0, 0))

    @skip
    def test_hourly_weather_report_no_summary(self):
        report = wthr.weather_report(self.wrequest_ll, summarize='none', kwargs=self.cache_kwargs)
        self.assertTrue(report is not None)
        self.assertTrue(isinstance(report, dict))
        self.assertTrue(
            all(k in ['report', 'hits', 'misses'] for k in report.keys())
        )
        self.assertTrue(isinstance(report.get('report'), pd.DataFrame))
        self.assertTrue(report.get('report').shape is not (0, 0))
