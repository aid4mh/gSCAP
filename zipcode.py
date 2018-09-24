#!/usr/bin/env python

""" A collection of scripts for processing weather data """

from collections import namedtuple
from contextlib import contextmanager
import datetime as dt
import io
import multiprocessing as mul
import os
from pathlib import Path
import re
import requests
import sys
from urllib3.exceptions import NewConnectionError
import time
import zipfile

import numpy as np
import pandas as pd


""" store the db cache in the users home directory """
home = str(Path.home())
path = os.path.join(home, '.brighten')
del home

if not os.path.exists(path):
    os.mkdir(path)

""" make sure the entire module is visible """
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'




if __name__ == '__main__':
    pass
