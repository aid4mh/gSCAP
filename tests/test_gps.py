import datetime as dt
import json
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

from gscapl import gps


class TestGPS(TestCase):
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

        cls.gcname = 'sqlite+pysqlite:///test_gps_cache.sqlite'
        cls.del_cache()

        engine = create_engine(cls.gcname, module=sqlite)
        gps.Base.metadata.create_all(engine)
        cls.session = sessionmaker(bind=engine)()

        cls.cache_kwargs = {'engine': cls.gcname}

        gps.CONNECTION_RESET_ATTEMPTS = 1
        gps.CONNECTION_WAIT_TIME = 0

        cls.home_cluster = cls.gen_cluster(0, 0, list(range(1, 7)) + list(range(18, 24)))
        cls.work_cluster = cls.gen_cluster(0.5, 0.5, list(range(8, 12)) + list(range(13, 17)))

        home_and_work = pd.concat(
                [cls.home_cluster, cls.work_cluster],
                sort=False)\
            .sort_values(by='ts')\
            .reset_index(drop=True)
        cls.home_and_work = gps.process_velocities(home_and_work)

    @classmethod
    def del_cache(cls):
        fn = cls.gcname.replace('sqlite+pysqlite:///', '')
        if os.path.exists(fn):
            os.remove(fn)

    @classmethod
    def mock_gmap_response(cls):
        fn = 'mock_gmap_response'
        if not os.path.exists(fn):
            fn = os.path.join('tests', fn)

        with open(fn, 'r') as f:
            mock_response = f.read()

        return mock_response

    @classmethod
    def gen_cluster(cls, lat, lon, hours):
        t = []
        for d in range(1, 7):
            for h in hours:
                for m in range(60):
                    t.append(dict(
                        ts=dt.datetime(
                            year=2019,
                            month=1,
                            day=d,
                            hour=h,
                            minute=m
                        ),
                        lat=lat+np.random.uniform(-0.0002, 0.0002),
                        lon=lon+np.random.uniform(-0.0002, 0.0002)
                    ))
        return pd.DataFrame(t, index=list(range(len(t))))

    @property
    def clusters(self):
        fn = 'some_clusters.csv'
        if not os.path.exists(fn):
            fn = os.path.join('tests', fn)

        return pd.read_csv(fn)

    @property
    def entries(self):
        fn = 'some_entries.csv'
        if not os.path.exists(fn):
            fn = os.path.join('tests', fn)

        df = pd.read_csv(fn, parse_dates=[
            'time_in', 'midpoint', 'time_out'
        ])
        df.duration = pd.to_timedelta(df.duration)
        return df

    @property
    def gps_records(self):
        fn = 'some_gps.csv'
        if not os.path.exists(fn):
            fn = os.path.join('tests', fn)

        return pd.read_csv(fn, parse_dates=['ts'])

    @classmethod
    def tearDownClass(cls):
        """perform when all tests are complete
        """
        cls.del_cache()

    def setUp(self):
        """perform before each unittest"""
        pass

    def tearDown(self):
        """perform after each unittest
        """
        t = self.session.query(gps.PlaceRequest).all()
        for ti in t:
            self.session.delete(ti)

        self.session.commit()

        responses.reset()

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

    @responses.activate
    def test_yelp_call(self):
        base = 'https://api.yelp.com/v3/businesses/search'
        url = f'{base}?latitude={self.lat}&longitude={self.lon}&radius=50&sort_by=best_match'
        responses.add(
            responses.GET,
            url,
            body='{"businesses": [], "total": 0, "region": {"center": {"longitude": -84.90685, "latitude": 32.3788}}}',
            status=200
        )

        t = gps.yelp_call(gps.PlaceRequest(
            lat=self.lat,
            lon=self.lon,
            radius=50,
            rankby=gps.YelpRankBy.BEST_MATCH
        ))

        self.assertTrue(isinstance(t, gps.PlaceRequest))

        j = json.loads(t.content)
        self.assertTrue(isinstance(j, dict))
        self.assertTrue('businesses' in j.keys())

    def test_parse_yelp_response(self):
        self.assertRaises(TypeError, gps.parse_yelp_response, 1)

        t = gps.parse_yelp_response('nan')
        self.assertTrue({'name', 'rank_order', 'categories', 'major_categories'} == set(t.keys()))
        self.assertTrue(t['name'] == 'not found')
        self.assertTrue(t['rank_order'] == -1)
        self.assertTrue(t['major_categories'] == 'none')

        t = gps.parse_yelp_response('}{')
        self.assertTrue(t['major_categories'] == 'JSONDecodeError')

        responses.reset()
        t = json.dumps(dict(
            businesses=[
                dict(
                    name='test',
                    categories=[
                        dict(alias='3dprinting')
                    ]
                )
        ]))
        t = gps.parse_yelp_response(t)
        self.assertTrue(t['name'] == 'test')
        self.assertTrue(t['major_categories'] == 'personal_services')

    @responses.activate
    def test_gmap_call(self):
        url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=32.3788%2C-84.90685&maxprice' \
              '=None&minprice=None&radius=50&rankby=prominence&key=AIza'
        responses.add(
            responses.GET,
            url=url,
            body=self.mock_gmap_response()
        )

        r = gps.PlaceRequest(
            lat=self.lat,
            lon=self.lon,
            radius=50,
            rankby=gps.GmapsRankBy.PROMINENCE
        )
        t = gps.gmap_call(r)
        self.assertTrue(isinstance(t, gps.PlaceRequest))

    def test_gmap_response(self):
        c = self.mock_gmap_response()
        c = gps.parse_gmap_response(c)

        self.assertTrue(c['rank_order'] == 0)
        self.assertTrue(c['name'] == 'c')
        self.assertTrue(c['categories'] == 'campground')
        self.assertTrue(c['major_categories'] == 'lodging')

    def test_gmapping(self):
        t = gps.gmapping('campground')
        self.assertTrue(t == {'lodging'})

        t = gps.gmapping(pd.Series(['campground']))
        self.assertTrue(t == {'lodging'})

        t = gps.gmapping('Expecting value: d')
        self.assertTrue(t == {'JSON Decode Error'})

        t = gps.gmapping('.')
        self.assertTrue(t  == {'undefined category'})

    @responses.activate
    def test_process_request(self):
        request = gps.PlaceRequest(
            lat=self.lat,
            lon=self.lon,
            source=gps.ApiSource.YELP,
            rankby=gps.YelpRankBy.BEST_MATCH,
            radius=50
        )
        progqu, reqque, resque = mul.Queue(), mul.Queue(), mul.Queue()
        args = (request, True, False, progqu, reqque, resque)

        resque.put(dict(
            pid=os.getpid(),
            response=request
        ))

        t = gps.process_request(args)
        self.assertTrue(t['hits'] == 1 and t['misses'] == 0)
        self.assertTrue(isinstance(t['report'], dict))

        t = pd.DataFrame(t['report'], index=[0])
        self.assertTrue(isinstance(t, pd.DataFrame))

        resque.put(dict(
            pid=os.getpid(),
            response=None
        ))
        args = (request, False, False, progqu, reqque, resque)
        self.assertRaises(
            ConnectionError, gps.process_request, args
        )

        base = 'https://api.yelp.com/v3/businesses/search'
        url = f'{base}?latitude={self.lat}&longitude={self.lon}&radius=50&sort_by=best_match'
        responses.add(
            responses.GET,
            url,
            body='{"businesses": [], "total": 0, "region": {"center": {"longitude": -84.90685, "latitude": 32.3788}}}',
            status=200
        )
        resque.put(dict(
            pid=os.getpid(),
            response=None
        ))
        args = (request, False, False, progqu, reqque, resque)
        t = gps.process_request(args)
        self.assertTrue(t['hits'] == 0)
        self.assertTrue(t['misses'] == 1)

        args = (request, True, False, progqu, reqque, resque)
        resque.put(dict(
            pid=os.getpid(),
            response=None
        ))
        t = gps.process_request(args)
        self.assertTrue(t['report']['content'] == '{"error": "not found in cache"}')

        def empty_and_close(qu):
            while not qu.empty():
                qu.get()
            qu.close()

        empty_and_close(progqu)
        empty_and_close(reqque)
        empty_and_close(resque)

        del progqu, reqque, resque

    @responses.activate
    def test_request_nearby_places(self):
        base = 'https://api.yelp.com/v3/businesses/search'
        url = f'{base}?latitude={self.lat}&longitude={self.lon}&radius=50&sort_by=best_match'
        responses.add(
            responses.GET,
            url,
            body='{"businesses": [], "total": 0, "region": {"center": {"longitude": -84.90685, "latitude": 32.3788}}}',
            status=200
        )

        # check a single request
        request = gps.PlaceRequest(
            lat=self.lat,
            lon=self.lon,
            source=gps.ApiSource.YELP,
            rankby=gps.YelpRankBy.BEST_MATCH,
            radius=50
        )
        t = gps.request_nearby_places(request, 1, kwargs=self.cache_kwargs)
        self.assertTrue(isinstance(t['request'], pd.DataFrame))
        self.assertTrue(t['misses'] == 1 and t['hits'] == 0)
        self.assertTrue(len(t['request']) == 1)

        # check multiple
        request = [request for i in range(2)]
        t = gps.request_nearby_places(request, 1, kwargs=self.cache_kwargs)
        self.assertTrue(isinstance(t['request'], pd.DataFrame))
        self.assertTrue(t['misses'] == 0 and t['hits'] == 2)
        self.assertTrue(len(t['request']) == 2)

    def test_update_qu(self):
        qu = mul.Queue()
        gps.update_queue(qu)
        t = qu.get()
        self.assertTrue(t == 1)

    def test_get_from_cache(self):
        t = gps.PlaceRequest(
            lat=self.lat,
            lon=self.lon,
            source=gps.ApiSource.YELP,
            rankby=gps.YelpRankBy.BEST_MATCH,
            radius=50
        )
        self.session.add(t)
        self.session.commit()

        t = gps.get_from_cache(t, self.session)
        self.assertTrue(t is not None)
        self.assertTrue(isinstance(t, gps.PlaceRequest))

    def test_put_to_cache(self):
        t = gps.PlaceRequest(
            lat=self.lat,
            lon=self.lon,
            source=gps.ApiSource.YELP,
            rankby=gps.YelpRankBy.BEST_MATCH,
            radius=50
        )
        gps.put_to_cache(t, self.session)

        t = self.session.query(gps.PlaceRequest).all()
        self.assertTrue(len(t) == 1)

    def test_cache_man(self):
        reqqu, resqu = mul.Queue(), mul.Queue()

        # test the get
        reqqu.put(dict(
            pid=os.getpid(),
            type='get',
            args=[gps.PlaceRequest(
                lat=self.lat,
                lon=self.lon,
                source=gps.ApiSource.YELP,
                rankby=gps.YelpRankBy.BEST_MATCH,
                radius=50
            )]
        ))
        reqqu.put(dict(type='end'))
        gps.cache_manager(reqqu, resqu, self.gcname)

        t = resqu.get()
        self.assertTrue(t['response'] is None)

        # test the put
        reqqu.put(dict(
            pid=os.getpid(),
            type='put',
            args=[gps.PlaceRequest(
                lat=self.lat,
                lon=self.lon,
                source=gps.ApiSource.YELP,
                rankby=gps.YelpRankBy.BEST_MATCH,
                radius=50
            )]
        ))
        reqqu.put(dict(type='end'))
        gps.cache_manager(reqqu, resqu, self.gcname)

        t = self.session.query(gps.PlaceRequest).all()
        self.assertTrue(len(t) == 1)
        self.assertTrue(t[0].content is None)

    def test_api_source(self):
        self.assertTrue(gps.api_source('Google Places') == gps.ApiSource.GMAPS)
        self.assertTrue(gps.api_source('Yelp') == gps.ApiSource.YELP)
        self.assertRaises(KeyError, gps.api_source, 'none')

    def test_cluster_metrics(self):
        t = gps.cluster_metrics(self.clusters, self.entries)
        self.assertTrue(
            list(t.columns) == [
                'username', 'cid', 'name', 'lat', 'lon', 'categories',
                'max_duration', 'mean_duration', 'mean_ti_between_visits',
                'min_duration', 'std_duration', 'times_entered',
                'total_duration'
            ]
        )
        self.assertTrue(isinstance(t, pd.DataFrame))
        self.assertTrue('xNot' not in t.cid)

    def test_process_velocities(self):
        t = gps.process_velocities(self.gps_records.iloc[:2])
        self.assertTrue(
            set(t.columns) == {
                'ts', 'lat', 'lon', 'binning', 'displacement',
                'time_delta', 'velocity'
            }
        )
        self.assertTrue(t.loc[1].binning == 'stationary')
        self.assertTrue(t.loc[1].displacement == 11.1)
        self.assertTrue(t.loc[1].time_delta == 60)
        self.assertTrue(t.loc[1].velocity == 0.185)

    def test_vdiscrete_powered(self):
        start = dt.datetime(year=2018, month=1, day=1)
        end = start + dt.timedelta(minutes=1)

        result = gps.discrete_velocity(
            (47.679853, -122.325744, start), (47.673600, -122.364783, end)
        )

        self.assertTrue(result.get('binning') == 'powered_vehicle')

    def test_vdiscrete_walking(self):
        start = dt.datetime(year=2018, month=1, day=1)
        end = start + dt.timedelta(hours=1)

        result = gps.discrete_velocity(
            (47.679853, -122.325744, start), (47.673600, -122.364783, end)
        )

        self.assertTrue(result.get('binning') == 'walking')

    def test_vdiscrete_stationary(self):
        start = dt.datetime(year=2018, month=1, day=1)
        end = start + dt.timedelta(hours=1)

        result = gps.discrete_velocity(
            (47.679853, -122.325744, start), (47.679853, -122.325744, end)
        )

        self.assertTrue(result.get('binning') == 'stationary')

    def test_vdiscrete_brunch(self):
        start = dt.datetime(year=2018, month=1, day=1)
        end = start + dt.timedelta(minutes=30)

        result = gps.discrete_velocity(
            (47.679853, -122.325744, start), (47.673600, -122.364783, end)
        )

        self.assertTrue(result.get('binning') == 'active')

    def test_vdiscrete_high_speed(self):
        start = dt.datetime(year=2018, month=1, day=1)
        end = start + dt.timedelta(hours=2)

        result = gps.discrete_velocity(
            (47.679853, -122.325744, start), (40.772849, -111.838413, end)
        )

        self.assertTrue(result.get('binning') == 'high_speed_transportation')

    def test_vdiscrete_anomaly(self):
        start = dt.datetime(year=2018, month=1, day=1)
        end = start + dt.timedelta(minutes=1)

        result = gps.discrete_velocity(
            (47.679853, -122.325744, start), (40.772849, -111.838413, end)
        )

        self.assertTrue(result.get('binning') == 'anomaly')

    def test_vdiscrete_throws(self):
        self.assertRaises(TypeError, gps.discrete_velocity, (0, 0, 0), (0, 0, dt.datetime.now()))
        self.assertRaises(TypeError, gps.discrete_velocity, (0, 0, dt.datetime.now(), (0, 0, 0)))

    def test_estimate_home(self):
        t = gps.estimate_home_location(self.gps_records)
        self.assertTrue(t[0] is None and len(t[1]) == 0)

        r = pd.concat([
            self.gps_records for i in range(100)
        ], axis=0, sort=False)
        r['ts'] = dt.datetime(
            year=2005, month=1, day=1, hour=4, minute=4
        )
        t = gps.estimate_home_location(r)
        self.assertTrue(t[0]['cid'] == 'home')
        self.assertTrue(t[0]['lat'] == 40.00015)
        self.assertTrue(t[0]['lon'] == -45.0)

    def test_estimate_work(self):
        t = gps.estimate_work_location(self.gps_records)
        self.assertTrue(t[0] is None and len(t[1]) == 0)

        r = pd.concat([
            self.gps_records for i in range(100)
        ], axis=0, sort=False)
        r['ts'] = dt.datetime(
            year=2005, month=1, day=3, hour=12, minute=4
        )
        t = gps.estimate_work_location(r)
        self.assertTrue(t[0]['cid'] == 'work')
        self.assertTrue(t[0]['lat'] == 40.00015)
        self.assertTrue(t[0]['lon'] == -45.0)

    def test_geo_pairwise(self):
        x = [(0, 0), (1, 0)]
        t = gps.geo_pairwise_distances(x)
        self.assertTrue(len(t) == 1)
        self.assertTrue(t[0] == 111194.9)

        x.append((0, 1))
        t = gps.geo_pairwise_distances(x)
        self.assertTrue(len(t) == 3)

    def test_get_clusters_with_context(self):
        records, clusters = gps.get_clusters_with_context(self.home_and_work)
        clusters = clusters.cid.unique()
        self.assertTrue('work' in clusters)
        self.assertTrue('home' in clusters)

    def test_get_clusters_with_context_only_home_when_work_out_of_range(self):
        work = self.work_cluster.copy()
        work.lat = work.lat + 10

        y = pd.concat([self.home_cluster, work], sort=False).sort_values(by='ts').reset_index(drop=True)
        y = gps.process_velocities(y)

        records, clusters = gps.get_clusters_with_context(y)

        clusters = clusters.cid.unique()
        self.assertTrue('work' not in clusters)
        self.assertTrue('home' in clusters)

    def test_get_clusters_with_context_only_home_when_not_working(self):
        y = self.home_and_work.copy()
        y['working'] = False

        records, clusters = gps.get_clusters_with_context(y)
        clusters = clusters.cid.unique()
        self.assertTrue('work' not in clusters)
        self.assertTrue('home' in clusters)
