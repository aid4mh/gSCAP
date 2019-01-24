![Build Status](https://travis-ci.com/UW-Creativ-Lab/gSCAP.svg?branch=master)

# Geospatial Context Analysis Pipeline (gSCAP)

## Installation and Configuration
The module can be installed from the source repo using pip  
  
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