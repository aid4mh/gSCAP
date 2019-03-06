![Build Status](https://travis-ci.com/UW-Creativ-Lab/gSCAP.svg?branch=master)

# Geospatial Context Analysis Pipeline (gSCAP)
Full documentation can be found in these [Github pages](https://uw-creativ-lab.github.io/gSCAP/).

## Installation and Configuration
Currently, the module is compatible with Python versions >= 3.6. It can be installed from the source repo using pip  
  
 > `pip install git+https://github.com/UW-Creativ-Lab/gSCAP`

Several methods within the module use external APIs to extract semantic context. A configure file must be present
in the user's home directory titled '.gscapConfig' or loaded via `utils.load_config(file_path)`.  
The file should contain each API key on a separate line, separated by an equals sign as shown in the example below.  
  
  > GooglePlacesAPI=YourKey  
  > YelpAPI=YourKey  
  > DarkSkyAPI=YourKey  

More information for each API can be found at the following links:  
  * Google Places: https://developers.google.com/places
  * Yelp: https://www.yelp.com/fusion
  * DarkSky: https://darksky.net/dev

## Use Cases and Examples
The following sections show use cases and examples for both the GPS and Weather modules. However, demos have been prepared using Jupyter notebooks and may be viewed at [demos](https://github.com/UW-Creativ-Lab/gSCAP/tree/master/notebooks/demos).

### GPS Data ([demo](https://github.com/UW-Creativ-Lab/gSCAP/blob/master/notebooks/demos/gps.ipynb))
The major use cases of this package provide clustering mechanisms for GPS data. For this, we use a density based approach to unsupervised clustering: DBSCAN. Density based clustering algorithms are great candidates for GPS data as it is inherently noisy with clusters forming wherever stationary. However, picking appropriate hyperparameters across a cohort of participants using a wide array of devices can be difficult. In our study, we had devices that undersampled and devices that oversampled leading to inconsistent sampling and location densities across the cohort. The method we chose to help alleviate the issue is to over or under sample streams depending on the embedded frequency of points. 

#### Down-sample the stream to a predefined frequency
Some devices collect far more points than others. Maybe they have different settings for location services or multiple applications are requesting coordinate. The following method resamples by mean values for each minute of points. Consider the following dataframe of points. This is an example where too many points were collected in a single minute and would likely lead to a higher-than-normal density at the given location.

|     lat |      lon | ts                         |
|---------|----------|----------------------------|
| 45.5047 | -122.783 | 2015-06-01 00:01:04.561000 |
| 45.5047 | -122.783 | 2015-06-01 00:01:08        |
| 45.5047 | -122.783 | 2015-06-01 00:01:10        |
| 45.5047 | -122.783 | 2015-06-01 00:01:12        |
| 45.5047 | -122.783 | 2015-06-01 00:01:15        |
| 45.5047 | -122.783 | 2015-06-01 00:02:16.037000 |
| 45.5047 | -122.783 | 2015-06-01 00:03:01.134000 |
| 45.5047 | -122.783 | 2015-06-01 00:07:51.359000 |
| 45.5047 | -122.783 | 2015-06-01 00:09:10.460000 |
| 45.5047 | -122.783 | 2015-06-01 00:10:12.570000 |

In order to systematically decrease oversampling you can use the following code as an example:

```python
df = gps.resample_gps_intervals(df)
```

The method returns the following dataframe containing the mean latitude and longitude values for each available minute in the provided dataframe.

|     lat |      lon | ts                  |
|---------|----------|---------------------|
| 45.5047 | -122.783 | 2015-06-01 00:01:00 |
| 45.5047 | -122.783 | 2015-06-01 00:02:00 |
| 45.5047 | -122.783 | 2015-06-01 00:03:00 |
| 45.5047 | -122.783 | 2015-06-01 00:07:00 |
| 45.5047 | -122.783 | 2015-06-01 00:09:00 |

#### Up-sample the stream to a defined frequency
To adjust for undersampling we can impute *stationary* coordinates between points. This is used to account for specific data collection methods that don't take samples if the research participant hasn't moved locations, has been under a roof for many hours, or any other reason a phone isn't reporting a fresh location. Imputation frequencies are defined using the standard Pandas definitions and are defined in the [Pandas Docs](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)

```Python
df = gps.impute_stationary_coordinates(df), freq='30S')
```

|      lat |        lon | ts                  | binning    |   displacement |   time_delta |   velocity |
|----------|------------|---------------------|------------|----------------|--------------|------------|
| 45.50471 | -122.78298 | 2015-06-01 00:01:00 | null       |      nan       |    nan       |  nan       |
| 45.50471 | -122.78298 | 2015-06-01 00:01:30 | stationary |        0.00000 |     30.00000 |    0.00000 |
| 45.50473 | -122.78290 | 2015-06-01 00:02:00 | stationary |        6.80000 |     30.00000 |    0.22600 |
| 45.50473 | -122.78290 | 2015-06-01 00:02:30 | stationary |        0.00000 |     30.00000 |    0.00000 |
| 45.50471 | -122.78284 | 2015-06-01 00:03:00 | stationary |        5.20000 |     30.00000 |    0.17300 |
| 45.50471 | -122.78284 | 2015-06-01 00:03:30 | stationary |        0.00000 |     30.00000 |    0.00000 |
| 45.50471 | -122.78284 | 2015-06-01 00:04:00 | stationary |        0.00000 |     30.00000 |    0.00000 |
| 45.50471 | -122.78284 | 2015-06-01 00:04:30 | stationary |        0.00000 |     30.00000 |    0.00000 |
| 45.50471 | -122.78284 | 2015-06-01 00:05:00 | stationary |        0.00000 |     30.00000 |    0.00000 |
| 45.50471 | -122.78284 | 2015-06-01 00:05:30 | stationary |        0.00000 |     30.00000 |    0.00000 |

As you can see, the method above computes displacement, time, velocity, and estimates a velocity bin for each point. This is also a valuable tool and can be used directly with `gps.process_velocities(df)`

### Clustering
This method partitions the records into three sets - home, work, and everything else. Then, performs separate clustering on each partition. Two variables are returned. The first is the records with an additional row assigning cluster cids, and the second is the clusters themselves.

This function utilizes the unsupervised clustering algorithm DBSCAN and the provided records. Parameters should be a dict and contain both min_samples and eps. See the scikit-learn docs for more information. Default parameters will be used if none are supplied. However, these parameters were tuned to generalize well on a specific dataset and I recommend retuning those paramaters to the dataset being used.

```python
df, clusters = gps.get_clusters_with_context(df)
```

The altered GPS records.

|     lat |      lon | ts                  | binning    |   displacement |   time_delta |   velocity | cid   |   distance_from_home |
|---------|----------|---------------------|------------|----------------|--------------|------------|-------|----------------------|
| 45.5047 | -122.783 | 2015-06-01 00:01:00 | null       |          nan   |          nan |    nan     | xNot  |              8.96678 |
| 45.5047 | -122.783 | 2015-06-01 00:02:00 | stationary |            6.8 |           60 |      0.113 | home  |              4.51555 |
| 45.5047 | -122.783 | 2015-06-01 00:03:00 | stationary |            5.2 |           60 |      0.086 | home  |              8.61803 |
| 45.5047 | -122.783 | 2015-06-01 00:07:00 | stationary |            0   |          240 |      0     | home  |              8.61803 |
| 45.5047 | -122.783 | 2015-06-01 00:09:00 | stationary |            5.9 |          120 |      0.049 | home  |              8.89559 |

The table of clusters.

| cid   |     lat |      lon | name   | categories   |
|-------|---------|----------|--------|--------------|
| home  | 45.5048 | -122.783 | home   | home         |
| work  | 45.5116 | -122.685 | work   | work         |
| x0    | 45.5122 | -122.684 | nap    | nap          |
| x1    | 45.5483 | -122.651 | nap    | nap          |

Here we see 'nap' in the name and categories columns of clusters. At this point, we haven't yet called Google Places or Yelp to get the semantic context of the cluster. To do this, we use the method shown in the following example.

```python
xone = list(clusters.loc[clusters.cid=='x0'].itertuples())[0]

request = gps.PlaceRequest(
    lat=xone.lat,
    lon=xone.lon, 
    radius=50, 
    source=gps.ApiSource.YELP,
    rankby=gps.YelpRankBy.BEST_MATCH,
)

results = gps.request_nearby_places(request)
```
A transposed version of the results looks like:

| cols             | vals                          |
|------------------|-------------------------------|
| dtRetrieved      | 2019-03-06 12:30:37.329597    |
| lat              | 45.51224                      |
| lon              | -122.68435                    |
| radius           | 50                            |
| source           | Yelp                          |
| name             | Basha's Mediterranean Cuisine |
| rank_order       | 0.0                           |
| categories       | mideastern, foodstands        |
| major_categories | dining_out                    |

### Weather lookups ([demo](https://github.com/UW-Creativ-Lab/gSCAP/blob/master/notebooks/demos/weather.ipynb))
The weather module allows you to lookup an hourly or daily weather summary for a given coordinate pair (or zipcode) and datetime. See the demo for more information and uses.

```python
now = dt.datetime.now()
today = dt.datetime(year=now.year, month=now.month, day=now.day); today

report = weather.weather_report((47.6062, 122.3321, today))
pd.DataFrame(report.get('report')).head()
```

A transposed version of the response 

| col                | val                 |
|--------------------|---------------------|
| cloud_cover_IQR    | 0.0                 |
| cloud_cover_mean   | 0.01                |
| cloud_cover_median | 0.0                 |
| cloud_cover_std    | 0.02                |
| date               | 2019-03-06 00:00:00 |
| dew_point_IQR      | 9.04                |
| dew_point_mean     | 6.02                |
| dew_point_median   | 6.04                |
| dew_point_std      | 4.63                |
| humidity_IQR       | 0.18                |
| humidity_mean      | 0.34                |
| humidity_median    | 0.35                |
| humidity_std       | 0.11                |
| lat                | 47.6062             |
| lon                | 122.3321            |
| precip_sum         | 0                   |
| temp_IQR           | 16.53               |
| temp_mean          | 32.22               |
| temp_med           | 29.61               |
| temp_std           | 9.88                |
| zipcode            | 4631                |

## Issues and enhancements
For bugs or feature requests please fill out the respective [ticket](https://github.com/UW-Creativ-Lab/gSCAP/issues/new/choose).