import datetime as dt
from unittest import TestCase

import pandas as pd

from gscapl import utils


class TestGPS(TestCase):
    """test class for
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization

        """
        now = dt.datetime.now()
        cls.day = dt.datetime(year=now.year, month=now.month, day=now.day)
        cls.lat = 43.7
        cls.lon = -64.8
        cls.zipcode = 47579

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
        results = utils.zip_from_dd(self.lat, self.lon, maxd=1)
        self.assertTrue(results == -1)

    def test_zip_from_dd_returns_correct(self):
        results = utils.zip_from_dd(self.lat, self.lon)
        self.assertTrue(results == 4631)