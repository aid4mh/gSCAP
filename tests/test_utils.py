import datetime as dt
from io import StringIO
import multiprocessing as mul
import os
import sys
from unittest import TestCase
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from gscap import utils


class TestGPS(TestCase):
    """test class for
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization

        """
        now = dt.datetime.now()
        cls.day = dt.datetime(year=now.year, month=now.month, day=now.day)
        cls.lat = 38.11094
        cls.lon = -86.91513
        cls.zipcode = 47579

    @classmethod
    def capture_out(cls, func, args):
        capture = StringIO()
        sys.stdout = capture

        func(*args)

        sys.stdout = sys.__stdout__
        out = capture.getvalue()

        capture.close()
        del capture
        return out

    @classmethod
    def tearDownClass(cls):
        """perform when all tests are complete
        """
        pass

    def setUp(self):
        """perform before each unittest"""
        pass

    def tearDown(self):
        """perform after each unittest
        """
        pass

    def test_zip_from_dd_returns_neg_when_no_results(self):
        results = utils.zip_from_dd(22, 122, maxd=1)
        self.assertTrue(results == -1)

    def test_zip_from_dd_returns_correct(self):
        results = utils.zip_from_dd(self.lat, self.lon)
        self.assertTrue(results == 47579)

    def test_isint(self):
        self.assertTrue(all(
            utils.isint(i) for i in [
                1, 1.0, '1', '1.0', -1, '-1.0'
            ]
        ))
        self.assertTrue(all(
            not utils.isint(i) for i in [
                'a', 'b', '1.a', object
            ]
        ))

    def test_isfloat(self):
        self.assertTrue(all(
            utils.isfloat(i) for i in [
                1, 1.0, '1', '1.0', -1, '-1.0'
            ]
        ))

        self.assertTrue(all(
            not utils.isfloat(i) for i in [
                'a', 'b', '1.a', object
            ]
        ))

    def test_zipcode_type_check(self):
        self.assertTrue(all([
            isinstance(utils.check_zipcode_type(2345), int),
            isinstance(utils.check_zipcode_type('2345'), int)
        ]))
        self.assertRaises(TypeError, utils.check_zipcode_type, 'a45')
        self.assertRaises(ValueError, utils.check_zipcode_type, -1)
        self.assertRaises(ValueError, utils.check_zipcode_type, '-1.')

    def test_dd_from_zip(self):
        lat, lon = utils.dd_from_zip(self.zipcode)
        self.assertTrue(lat == self.lat and lon == self.lon)

        lat, lon = utils.dd_from_zip(1)
        self.assertTrue(lat == 0 and lon == 0)

    def test_zip_from_dd(self):
        zipc = utils.zip_from_dd(self.lat, self.lon)
        self.assertTrue(zipc == self.zipcode)

        zipc = utils.zip_from_dd(self.lat, self.lon)
        self.assertTrue(zipc == self.zipcode)

        message = self.capture_out(utils.zip_from_dd, (20, 142)).strip(os.linesep)
        self.assertTrue(message == 'WARNING: closest zipcode found was 210.5Km from (20, 142)')

        message = self.capture_out(utils.zip_from_dd, (20, 142, 1)).strip(os.linesep)
        self.assertTrue(message == 'WARNING: zipcode not found within 1Km of (20, 142)')

        self.assertTrue(utils.zip_from_dd(20, 142, 1) == -1)

    def test_latlon_range_check(self):
        self.assertRaises(
            ValueError, utils.lat_lon_range_check, -91, 0
        )
        self.assertRaises(
            ValueError, utils.lat_lon_range_check, 91, 0
        )
        self.assertRaises(
            ValueError, utils.lat_lon_range_check, -181, 0
        )
        self.assertRaises(
            ValueError, utils.lat_lon_range_check, -0, 181
        )
        self.assertTrue(utils.lat_lon_range_check(0, 0) is None)

    def test_tz_from_dd(self):
        tz = utils.tz_from_dd((self.lat, self.lon))
        self.assertTrue(tz == ['America/Chicago'])

        tz = utils.tz_from_dd(
            [(self.lat, self.lon) for i in range(2)]
        )
        self.assertTrue(all(tzi == 'America/Chicago' for tzi in tz))

        points = pd.DataFrame(
            [[self.lat, self.lon], [self.lat, self.lon]],
            columns=['lat', 'lon']
        )
        tz = utils.tz_from_dd(points)
        self.assertTrue(all(tzi == 'America/Chicago' for tzi in tz))

    def test_tz_from_zip(self):
        tz = utils.tz_from_zip(self.zipcode)
        self.assertTrue(tz == ['America/Chicago'])

        tz = utils.tz_from_zip([self.zipcode, self.zipcode])
        self.assertTrue(all(tzi == 'America/Chicago' for tzi in tz))

        tz = utils.tz_from_zip(pd.Series([self.zipcode, self.zipcode]))
        self.assertTrue(all(tzi == 'America/Chicago' for tzi in tz))

    def test_geo_distance(self):
        self.assertTrue(
            utils.geo_distance(self.lat, self.lon, self.lat, self.lon) == 0
        )
        d = utils.geo_distance(self.lat, self.lon, self.lat+1, self.lon+1)
        self.assertTrue(np.isclose(d, 141114.06626067968))
