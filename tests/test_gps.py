import datetime as dt
from unittest import skip, TestCase

import pandas as pd

from gscapl import gps


class TestGPS(TestCase):
    """test class for
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization

        """
        pass

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

    def test_gpsrecords_named_tuple_conversion(self):
        d = dt.datetime(year=2019, month=1, day=1)
        gpsr = [
            gps.GPS(1, 1, d)
        ]
        rec = gps.gpsr_to_records(gpsr)

        assert isinstance(rec, pd.DataFrame)
        assert len(rec) == 1
        assert all(c in rec.columns for c in [
            'lat', 'lon', 'ts'
        ])

        rec = rec.iloc[0]
        assert rec.lat == 1
        assert rec.lon == 1
        assert rec.ts == d

    def test_records_to_gpsr(self):
        r = pd.DataFrame(columns=['lat', 'lon', 'ts'])
        d = dt.datetime(year=2019, month=1, day=1)
        r.loc[0] = (1, 1, d)

        gpsr = gps.records_to_gpsr(r)

        assert isinstance(gpsr, list)
        assert len(gpsr) == 1
        assert isinstance(gpsr[0], gps.GPS)

    def test_gps_dbscan_accepts_both_types(self):
        r = pd.DataFrame(columns=['lat', 'lon', 'ts'])
        d = dt.datetime(year=2019, month=1, day=1)
        r.loc[0] = (1, 1, d)

        l, c = gps.gps_dbscan(r)
        assert isinstance(l, list) and isinstance(c, list)
        assert len(l) == 1 and len(c) == 0

        r = gps.records_to_gpsr(r)
        l, c = gps.gps_dbscan(r)
        assert isinstance(l, list) and isinstance(c, list)
        assert len(l) == 1 and len(c) == 0

    @skip
    def test_takeout_parser(self):
        fn = 'location_history.json'
        results = gps.prep_takeout_data(fn)
        self.assertTrue(isinstance(results, pd.DataFrame))