# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:43:17 2023

@author: yunus
"""
import geopandas as gpd
from shapely.geometry.polygon import Polygon

#generalize this
#user inputs: State, city, and coordinates
name = "Greenville"
state = "Mississippi"
ymax = 33.457949
xmin = -91.139097
ymin = 33.336130
xmax = -91.013098
######
city_limits = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

filename = "https://usbuildingdata.blob.core.windows.net/usbuildings-v2/" + state +  ".geojson.zip"

file = gpd.read_file(filename)
clip_gdf = gpd.clip(file, city_limits)

#get all these coordinates on a csv

export_file_name = "Centralized_elevcluster_" + name + ".txt"
counter = 0

f = open(export_file_name, "w")
f.write("buildings,V1,V2\n")
for i in clip_gdf.geometry:
    counter += 1
    b_name = "B" + str(counter)
    f.write(b_name + "," + str(i.centroid.x) + "," + str(i.centroid.y) + "\n")

f.close()
