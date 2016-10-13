#!/usr/bin/env python
# export PYTHONPATH=$PYTHONPATH:/Library/Frameworks/GDAL.framework/Versions/1.11/Python/2.7/site-packages
import sys
import argparse
import json

from geoip import geolite2
from geoip import open_database
from sklearn.externals import joblib
import pycountry
db = open_database('/Users/gen/Downloads/GeoLite2-City.mmdb') 

# formatted file: list where each element is a tuple for one worker, represented by (num_hits, ip)
worker_ips = joblib.load('cocottributes_worker_ips.jbl')

hist = {}
hist ['all data'] = []
hist['countries'] = {}
hist['timezone'] = {}
hist['countries']['num_hits'] = []
hist['countries']['num_workers'] = []
hist['countries']['items'] = []
hist['timezone']['num_hits'] = []
hist['timezone']['num_workers'] = []
hist['timezone']['items'] = []
map_out = [] # {"city_name": "city", "lat": "lat", "long": "long", "nb_visits": int(num_hits)}

for num_hits,ip in worker_ips:
    if ip == '' or ip == '0' or ip is None:
        continue
    match = db.lookup(ip)
    
    if match is None:
        continue

    country = match.country    
    local = match.timezone

    hist['all data'].append((int(num_hits), match.location,local,country))
    if country not in hist['countries']['items']:
        hist['countries']['items'].append(country)
        hist['countries']['num_hits'].append(num_hits)
        hist['countries']['num_workers'].append(1)
    else:
        idx = hist['countries']['items'].index(country)
        hist['countries']['num_hits'][idx]+=num_hits
        hist['countries']['num_workers'][idx]+=1
        
    if local not in hist['timezone']['items']:
        hist['timezone']['items'].append(local)
        hist['timezone']['num_hits'].append(num_hits)
        hist['timezone']['num_workers'].append(1)
    else:
        idx = hist['timezone']['items'].index(local)
        hist['timezone']['num_hits'][idx]+=num_hits
        hist['timezone']['num_workers'][idx]+=1  
                        
    map_out.append({"city_name": match.timezone,
                    "lat": str(match.location[0]),
                    "long": str(match.location[1]),
                    "nb_visits": int(num_hits)})  

with open('cocottributes.json','w') as data_file:    
    data = json.dump(map_out,data_file)
joblib.dump(hist, 'cocottributes_worker_locations.jbl', compress=6) 

for item in sorted(zip(hist['countries']['items'], 
                       hist['countries']['num_hits'], 
                       hist['countries']['num_workers']), key=lambda x:x[1], reverse=True):
    country = pycountry.countries.get(alpha2=item[0]).name
    print '{} & {} & {} \\\\'.format(country, item[1], item[2])
