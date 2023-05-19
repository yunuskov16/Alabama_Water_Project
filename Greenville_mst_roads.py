# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27, 2022

@author: yunus
"""
from sklearn.cluster import AgglomerativeClustering
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import pickle
import geopandas as gpd
import tifffile as tiff #needed for the tif data for perry county
#import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.spatial import distance_matrix
#from math import sin, cos, sqrt, atan2, radians
#import sys
from xlwt import Workbook
import shapely
from shapely.ops import snap, split, nearest_points, substring
#from shapely.geometry import MultiPoint, LineString
from shapely.geometry import Polygon, box, Point, MultiPoint, LineString, MultiLineString, GeometryCollection
#from dbfread import DBF
import osmnx as ox
import networkx as nx
import copy
import fiona

#import gr
#import pyrosm
#to get it working
#conda activate rioxarray_env
#import rasterio
#from rasterio.crs import CRS
#from rasterio.plot import plotting_extent
import rioxarray as rxr
#import earthpy.plot as ep
import rasterstats as rs
import os
import py3dep
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import xlwt
from xlwt import Workbook
import pyproj
from pyproj import Proj, transform
import osgeo as ogr

#uniontown boundaries -- UPDATE to include additional neighborhoods south of town
#boundaries have been updated to include southern neighborhoods
# xmin = -87.53#739617137372
# xmax = -87.47#841138757
# ymin = 32.42#32.42075991564647
# ymax = 32.48#87035515

# alternative bounding box, maybe helpful for troubleshooting
#xmin <- -87.5275
#xmax <- -87.4915
#ymin <- 32.4210
#ymax <- 32.45930

#bounding box for Donaldsville LA
# ymax = 30.113664
# xmin = -91.020590
# ymin = 30.072706
# xmax = -90.972202
#boundign box for Mississippi
ymax = 33.464#33.457949
xmin = -91.128#-91.139097
ymin = 33.336130
xmax = -91.015#-91.013098
######
# projections
#meter_epsg=32616 # this is UTM projection for North American zone 16N (includes Uniontown)
# def graphsetup(xn, xx, yn, yx):
#     #zooms out in the case that you have a graph that is unconnected
#     #the idea is that if you zoom out enough eventually everything will be connected
#     xn = xmin
#     xx = xmax
#     yn = ymin
#     yx = ymax
#     uniontown_bounds = [yx, yn, xx, xn]
#     #G=ox.graph_from_bbox(north=ymax, south=ymin, east=xmax, west=xmin, network_type='drive_service', simplify=True, retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
#     graph=ox.graph_from_bbox(*uniontown_bounds, simplify=False, retain_all=True, network_type='drive_service')
        
#     undir_g = graph.to_undirected()
#     while len(sorted(nx.connected_components(undir_g))) > 1:
#         xn += -0.01
#         xx += 0.01
#         yn += -0.01
#         yx += 0.01
#         uniontown_bounds = [yx, yn, xx, xn]
#         #G=ox.graph_from_bbox(north=ymax, south=ymin, east=xmax, west=xmin, network_type='drive_service', simplify=True, retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
#         graph=ox.graph_from_bbox(*uniontown_bounds, simplify=False, retain_all=True, network_type='drive_service')
#         undir_g = graph.to_undirected()
        
#     return graph

def readClusterFile(fileID):
    file = np.genfromtxt(fileID, delimiter=",", skip_header = 1)
    file = file[:,1:]
    return file

def haversinedist(lat1, lon1, lat2, lon2):
    R = 6373.0
    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance * 1000

# testing code to load roads and create graph in python
#G=ox.graph_from_place('Uniontown, Alabama',network_type='drive_service')
city_bounds = [ymax, ymin, xmax, xmin]
#G=ox.graph_from_bbox(north=ymax, south=ymin, east=xmax, west=xmin, network_type='drive_service', simplify=True, retain_all=False, truncate_by_edge=False, clean_periphery=True, custom_filter=None)
G=ox.graph_from_bbox(*city_bounds, simplify=False, retain_all=True, network_type='drive_service')
#G = graphsetup(xmin, xmax, ymin, ymax)
#fig, ax = ox.plot_graph(ox.project_graph(G))
# still need to conver the graph to undirected. 
G2 = ox.simplify_graph(G, strict = True, remove_rings = True)
fig, ax = ox.plot_graph(ox.project_graph(G2))

### save it as a shapefile for later
#ox.save_graph_shapefile(G, filepath='C:/Users/Sara_CWC/Documents/ANALYSIS/Uniontown_opt/', encoding='utf-8')
# if reloading from the saved shapefile
#nodes = gpd.read_file('C:/Users/Sara_CWC/Documents/ANALYSIS/Uniontown_opt/Review/All_12_2021/PythonFiles/Pythonv2_IP//nodes.shp')
#edges = gpd.read_file('C:/Users/Sara_CWC/Documents/ANALYSIS/Uniontown_opt/Review/All_12_2021/PythonFiles/Pythonv2_IP//edges.shp')
# Convert the graph to geodataframe - this is also the structure if you load the saved shapefile 
nodes = ox.graph_to_gdfs(G2, nodes=True, edges=False)
#edges = edges.dissolve()

# update the node names to string easier to deal with than numeric 

N_list_df={}
#nodes.insert(5,'n_id',None)

counter_N=0
for index, row in nodes.iterrows():
    nodes.loc[index,'n_id']= 'R'+str(counter_N)
    N_list_df[index]='R'+str(counter_N)
    counter_N+=1
G2 = nx.relabel_nodes(G2, N_list_df)

edges = ox.graph_to_gdfs(G2, nodes=False, edges=True)
nodes = ox.graph_to_gdfs(G2, nodes=True, edges=False)# resave with the correct road node names

multline_list = []
for i in edges.geometry:
    multline_list.append(i)

#check for any unconnected roads
G0 = G2.to_undirected()
largest_cc = max(nx.connected_components(G0), key=len)

nodes.insert(5,'n_id',list(G2.nodes))

#if you have an unconnected portion of the road we need to connect it if there is a road node with demand
#to find if a road node has wastewater demand we need to see if these nodes are closer
#to the building locations than the connected nodes
#step 1 find if there is a demand_node in the unconnected parts of the graph
unconnected_test0 = list(nx.connected_components(G0))
unconnected_dict = dict()
unconnected_test = []
for i in sorted(unconnected_test0, key = len, reverse = 1):
    unconnected_test.append(tuple(i))
unconnected_test = sorted(list(set(unconnected_test)), key = len, reverse= 1)

if len(unconnected_test) > 1:
    edge_connects = []
    graph_lists = unconnected_test[1:]
    list_multipoints = []
    for i in unconnected_test:
        main_network = []
        for j in i:
            x1, y1 = G0.nodes[j]['x'], G0.nodes[j]['y']
            point = Point(x1, y1)
            main_network.append(point)        
        main_network_multi = MultiPoint(main_network)
        list_multipoints.append(main_network_multi)
    
    connected = 0
    
    popped_multipoints = list_multipoints[1:]
    counter = 0
    largest_node_network = list_multipoints[0]
    while counter < len(unconnected_test)-1:
        distances = []
        names = []
        dist_name_dict = {}
        for i in popped_multipoints:
            near_geom = nearest_points(largest_node_network, i)
            p_in = nodes.loc[nodes['geometry'] == near_geom[0]]
            p_out = nodes.loc[nodes['geometry'] == near_geom[1]]
            p1 = str(p_in.index)
            p2 = str(p_out.index)
            p1_name = p1[p1.find('R'):p1.find(']')-1]
            p2_name = p2[p2.find('R'):p2.find(']')-1]
            line = LineString([near_geom[0], near_geom[1]])
            outProj = Proj(init='epsg:2163')
            inProj = Proj(init='epsg:4326')
            x1, y1 = transform(inProj, outProj, near_geom[0].x, near_geom[0].y)
            x2, y2 = transform(inProj, outProj, near_geom[1].x, near_geom[1].y)
            dist = sqrt((x1-x2)**2 + (y1-y2)**2)
            distances.append(dist)
            names.append((p1_name, p2_name, dist, line))
            dist_name_dict[dist] = (names[-1], i)
        min_dist = min(distances)
        if min_dist <= 250: 
            min_name = dist_name_dict[min_dist]
            edge_connects.append(min_name[0])
            popped_multipoints.pop(distances.index(min_dist))
            largest_node_network = largest_node_network.union(min_name[1])
                
        counter+= 1
        
#                 #code for connecting this entire graph to the largest_cc graph
#                 #do this the next time you open your laptop
#                 break
    # if connected <= 0:
    #     combine_unconnected_lists = set()
    #     for i in graph_lists:
    #         combine_unconnected_lists = combine_unconnected_lists.union(i)
    #     dropped_nodes = set()
    #     dropped_edges = set()
    #     for i in nodes.index.values:
    #         r_node = str(nodes['n_id'][i])
    #         if r_node in combine_unconnected_lists:
    #             dropped_nodes.add(r_node)
    #             relevant_edges_u = edges[edges['uN'] == r_node]
    #             for u in list(relevant_edges_u.index):
    #                 dropped_edges.add(u)
    #             relevant_edges_v = edges[edges['vN'] == r_node]
    #             for v in list(relevant_edges_v.index):
    #                 dropped_edges.add(v)
    #     nodes2 = nodes.drop(list(dropped_nodes))
    #     edges2 = edges.drop(list(dropped_edges))
counter = 1
for i, j, k, l in edge_connects:
    edges.loc[(i, j, 0), ['osmid', 'length', 'geometry']] = [counter, k, l]
    G0.add_edge(i, j, length = k)
    counter += 1

#edges = edges.to_crs("EPSG:4326")
#nodes = nodes.to_crs("EPSG:4326")
edges = edges.to_crs("EPSG:2163")
nodes = nodes.to_crs("EPSG:2163")




#pulling out the elevation for a given bounding box
def get_elevation_raster(bbox):
    y2, y1, x1, x2 = bbox
    p1 = Point(x1, y1)
    p2 = Point(x1, y2)
    p3 = Point(x2, y1)
    p4 = Point(x2, y2)
    geom = Polygon([p1, p2, p4, p3]) #where bbox is a polygon
    dem = py3dep.get_map("DEM", geom, resolution=10, geo_crs="epsg:4326", crs="epsg:3857")
    dem.rio.to_raster(os.path.realpath(os.path.join(os.path.dirname('Greenville_MS_Case'), '.')) + "\\city_dem.tif" )
    return os.path.realpath(os.path.join(os.path.dirname('Greenville_MS_case'), '.')) + "\\city_dem.tif"
    
dem_name = get_elevation_raster(city_bounds)
#save tif file for the R function and graphsetup

    
clusterfilename = 'Centralized_elevcluster_' + "Greenville" + '.txt'

# load the files for building locations 
clusterfile = os.path.realpath(os.path.join(os.path.dirname('Greenville_Case'), '.')) + '\\' + clusterfilename
building_coords = readClusterFile(clusterfile)
building_coords = building_coords[:, :2]
# load the building coordinate file
building_coords = pd.read_csv(clusterfile, index_col=0)
# convert to a shapefile to match nodes objective above - these will be merged into the graph
build_shp = gpd.GeoDataFrame(building_coords, 
            geometry=gpd.points_from_xy(building_coords.V1, building_coords.V2),
            crs="EPSG:4326")

build_shp=build_shp.to_crs("EPSG:2163")
build_shp.insert(3,'n_id',None)
counter_N=1
for index, row in build_shp.iterrows():
    build_shp.loc[index,'n_id']= 'B'+str(counter_N)
    counter_N+=1
    
#for each house connected to a demand node 
demand_dict = {}
for i in range(0, len(build_shp.geometry)):
    p = build_shp.geometry.iloc[i]
    pot_pts = nearest_points(build_shp.geometry.iloc[i], MultiPoint(list(nodes.geometry)))
    if len(pot_pts) == 1:
        node_row = nodes.loc[nodes['geometry'] == pot_pts[0]]
        if len(node_row) > 0:
            if list(node_row['n_id'])[0] not in demand_dict:
                demand_dict[list(node_row['n_id'])[0]] = 1
            else:
                demand_dict[list(node_row['n_id'])[0]] += 1
                
    else:
        node_row = nodes.loc[nodes['geometry'] == pot_pts[1]]        
        if list(node_row['n_id'])[0] not in demand_dict:
            demand_dict[list(node_row['n_id'])[0]] = 1
        else:
            demand_dict[list(node_row['n_id'])[0]] += 1
#go through each row and add the dictionary id to the corresponding dataframe
#0 if there is no houses that are connected to a given road node
n_demand = []
for i in nodes['n_id']:
    if i not in demand_dict:
        n_demand.append(0)
    else:
        n_demand.append(demand_dict[i])
        
nodes['n_demand'] = n_demand

term_nodes = []

for i in range(len(nodes)):
    name = nodes.iloc[i]['n_id']
    if nodes.iloc[i]['n_demand'] > 0:
        term_nodes.append(name)

G0 = G0.to_undirected()

#change to path distance
#iterating through each nodes
#use the shortest path algorithm from the graph set up code and take total length
#dist = lambda p1, p2: nx.shortest_path_length(G0, source = p1, target = p2, weight = 'length')
#test1 = np.asarray([[dist(p1, p2) for p1 in term_nodes] for p2 in term_nodes])
distances = dict(nx.all_pairs_dijkstra_path_length(G0, cutoff=None, weight='length'))
order = term_nodes
dist = np.zeros((len(order), len(order)))

for index1, member1 in enumerate(order):
    curr = distances.get(member1, {})
    for index2, member2 in enumerate(order):
        dist[index1, index2] = curr.get(member2, 0)
#don't delete eucledian
#test = [[float(nodes.loc[nodes['n_id'] == a].geometry.x), float(nodes.loc[nodes['n_id'] == a].geometry.y)] for a in term_nodes]
#test1 = distance_matrix(test, test)

#dist_df = pd.DataFrame(test1, columns = list(complete_df['n_id']), index = list(complete_df['n_id']))

#selected_data = test
#choose number of clusters with k
#note: we have changed from ward to complete linkage
ngroup = 20
clustering_model = AgglomerativeClustering(n_clusters=ngroup, affinity='precomputed', linkage='complete')
a = clustering_model.fit(dist)
b = clustering_model.labels_

n_clust = []


for i in range(len(nodes)):
    row = nodes.iloc[i]
    name = row['n_id']
    if name not in term_nodes:
        n_clust.append(-1)
    else:
        clust_idx = term_nodes.index(name)
        clust = b[clust_idx]
        n_clust.append(clust)
    
#add lonlat back 

term_nodes_dict = dict()
for i in range(0, ngroup):
    temp_node_list = []
    for j in range(len(term_nodes)):
        clust_num = b[j]
        if clust_num == i:
            temp_node_list.append(term_nodes[j])
    term_nodes_dict[i] = temp_node_list

#nx.algorithms.approximation.steinertree.metric_closure(G3.to_undirected())
mst_list = []
for i in range(0, ngroup):
    mst = nx.algorithms.approximation.steinertree.steiner_tree(G0, term_nodes_dict[i], weight = 'length')
    mst_list.append(mst)
    

n_clust = []
for i in nodes['n_id']:
    for j in range(len(mst_list)):
        if i in mst_list[j].nodes:
            n_clust.append(j)
            break
        if j == len(mst_list)-1:
            n_clust.append(-1)
nodes['cluster'] = n_clust
#### What to do now?
#we have nodes simplified and with assigned demands
#fix this
# G3=nx.MultiDiGraph()


# for i in range(len(nodes)):
#       G3.add_node(nodes['n_id'][i],x=nodes['geometry'][i].x,y=nodes['geometry'][i].y)
# for i, j, k in G2.edges:
#     G3.add_edge(i, j, weight=float(G2.edges[i, j, k]['length']),
#                 u=i,v=j)    
# G3 = G3.to_undirected()

def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


### use this function to determine cut the roads at points closest to where the buildings should intersect

def split_line_with_points(line, points):
    """Splits a line string in several segments considering a list of points.

    The points used to cut the line are assumed to be in the line string 
    and given in the order of appearance they have in the line string.

    >>> line = LineString( [(1,2), (8,7), (4,5), (2,4), (4,7), (8,5), (9,18), 
    ...        (1,2),(12,7),(4,5),(6,5),(4,9)] )
    >>> points = [Point(2,4), Point(9,18), Point(6,5)]
    >>> [str(s) for s in split_line_with_points(line, points)]
    ['LINESTRING (1 2, 8 7, 4 5, 2 4)', 'LINESTRING (2 4, 4 7, 8 5, 9 18)', 'LINESTRING (9 18, 1 2, 12 7, 4 5, 6 5)', 'LINESTRING (6 5, 4 9)']

    """
    segments = []
    current_line = line
    
    for p in points:
        d = current_line.project(p)
        if (d>0) & (d<current_line.length) : # check to make sure the points aren't too close to the ends points of the point
            seg, current_line = cut(current_line, d)
            segments.append(seg)
        else: 
            segments.append(current_line)
    return segments

# from https://github.com/ywnch/toolbox/blob/master/toolbox.py
# function to find the index of the nearest line 
def find_kne(point, lines):
    dists = np.array(list(map(lambda l: l.distance(point), lines.geometry))) # distance array of lines to points
    kne_pos = dists.argsort()[0] # returns the index that would sort the array
    kne = lines.iloc[[kne_pos]]
    kne_idx = kne.index[0]
    return kne_idx, kne.values[0]

#B_pts =[]
#Edges_diss = edges.dissolve() 
#Edges_diss = Edges_diss.geometry.unary_union# dissolve edsges to reduce the looping 
# goal here is to create a list of points where each building point should intersect the raod network.
# iterate through all the building points and split the roads into segments based on the intersections formed


# B_pts =[]
# for index, row in build_shp.iterrows():
#      point = row.geometry
#      idx,val = find_kne(point, edges) # find the nearest edge
#      l=edges[edges.index==idx].geometry.unary_union # geometry of the nearest edge
#      p_t = l.interpolate(l.project(point)) # project is the dstance along the line from the start point of the line, so interpolate is finding a point on the line at that distance
#      if p_t not in B_pts:
#          B_pts.append(p_t)

#B_nodes=gpd.GeoDataFrame(geometry=B_pts, crs="EPSG:2163")
#B_nodes.to_file('./B_nodes.shp')
#haversinedist(lat1, lon1, lat2, lon2)
#E_list=[]
#N_list=[] # this will have duplicates update to drop it after testing. 
nodes_out_li = []
edges_out_li = []
counter_N=len(G0.nodes)

for a in range(len(mst_list)):
    if len(mst_list[a]) == 0:
        nodes_out_li.append([])
        edges_out_li.append([])
        continue
    E_list_df={}
    N_list_df={}
    counter=0
    for E in range(0, len(mst_list[a].edges)):
        E_S = edges.iloc[E]
        #segments=[]
        distance_delta = 100 #100 # meters
        line=E_S.geometry
        node_names=[] # list for hte additional node names to be added 
        node_names.append(E_S.name[0])
        seg_final=[]
        #N_list.append(line.coords[0])
        # set the first point in the segment and add to the node list. 
        N_list_df[E_S.name[0]]={'n_id':E_S.name[0],'x':line.coords[0][0],'y':line.coords[0][1],
                                 'geometry':Point(line.coords[0])}
    
        # points=[]
        # for P in range(0,len(B_pts)):
        #     pt=B_pts[P]
        #     if line.intersects(pt) & (pt not in [Point(line.coords[0]),Point(line.coords[-1])]): # keep only the points taht are within the line segment
        #         points.append(pt)
        # if len(points)>0: # if a building point intersects first divide the segment at that point
        #     seg = split_line_with_points(line,points)
        #     for s in range(len(seg)): # then check if each subsegment is >100m if so split these again.
        #         seg_sub=seg[s]
        #         if seg_sub.length>100: 
        #               distances = np.arange(100, seg_sub.length, distance_delta) 
        #               #distances = np.append(distances, [[line.project(pp) for pp in points]]) 
        #               #distances=np.sort(distances)
        #             # define the list of points at which to split to the line
        #               pts = [seg_sub.interpolate(distance) for distance in distances]
        #               seg2 = split_line_with_points(seg_sub,pts)
        #               seg_final=seg_final+seg2
        #         else: # if the subsegment is less 100m add it to list without cutting
        #               seg_final=seg_final+[seg_sub]
        if line.length>100: # run the 100m segment code only if there is no building cut point to make        
                distances = np.arange(100, line.length, distance_delta) 
                #distances = np.append(distances, [[line.project(pp) for pp in points]]) 
                #distances=np.sort(distances)
            # define the list of points at which to split to the line
                pts = [line.interpolate(distance) for distance in distances]
                seg_final = split_line_with_points(line,pts)
                #seg, current_line = cut(line,distance=100)
                #segments.append(seg)
                # loop to add all nodes to the node list and
        # after the segment list has been finalized loop through it add the nodes and edges. 
        if len(seg_final)>0: # if there are segments that the edge needs to be split into add these 
            for j in range(1,len(seg_final)):
            
                        # skip the first segment since we already added the starting point
                t_n='R'+str(counter_N) # node name alays +1 the length of the existing node list. 
                node_names.append(t_n)
                        #N_list.append(seg[j].coords[0]) # add the first point of the segment the node list
                        # always add the first node from the segment to avoid duplication of points that join segments
                N_list_df[t_n]={'n_id':t_n,'x':seg_final[j].coords[0][0], 'y':seg_final[j].coords[0][1],
                                'geometry':Point(seg_final[j].coords[0])}
                counter_N+=1 # only update the node counter within the loop because end points already have names
            
            # finally add the last node to the list, it also already has a name, and is the end of the original line
            N_list_df[E_S.name[1]]={'n_id':E_S.name[1],'x':line.coords[-1][0],'y':line.coords[-1][1],
                                            'geometry':Point(line.coords[-1])} # pull the last point from the segment 
            node_names.append(E_S.name[1])
            # iterate through the segments again to create the edge list. 
            for j in range(0,len(seg_final)):
                E_list_df[counter]={'uN':node_names[j],'u':seg_final[j].coords[0], 'vN':node_names[j+1],
                                    'v':seg_final[j].coords[-1],'e_id':E, 'osmid':E_S.osmid,
                                    'geometry':seg_final[j]}
            
                counter+=1
                if seg_final[j].length >105:
                    print(node_names[j])
        else: # this is when the line length is less than 100m and there's no building point to cut and theefore the original line needs to be added
            N_list_df[E_S.name[1]]={'n_id':E_S.name[1],'x':line.coords[-1][0],'y':line.coords[-1][1],
                                    'geometry':Point(line.coords[-1])} # pull the last point from the segment 
            E_list_df[counter]={'uN':E_S.name[0], 'u':line.coords[0],'vN':E_S.name[1], 
                                'v':line.coords[-1],'e_id':E, 'osmid':E_S.osmid,
                                'geometry':line}
            counter+=1
            if line.length>105:
                print(E_S.name[0])

    nodes_out=gpd.GeoDataFrame(pd.DataFrame.from_dict(data=N_list_df, orient='index'), crs="EPSG:2163")
    edges_out = gpd.GeoDataFrame(pd.DataFrame.from_dict(data=E_list_df, orient='index'), crs="EPSG:2163")
    # add the length attribute
    #also convert the name value dtypes into strings
    nodes_out['n_id'] = nodes_out['n_id'].values.astype(str)
    edges_out['uN'] = edges_out['uN'].values.astype(str)
    edges_out['vN'] = edges_out['vN'].values.astype(str)
    
    edges_out['length']=edges_out.geometry.length
    ## for testing
    edges_out[['length','uN','vN','geometry']].to_file('./edges_out.shp')
    nodes_out[['n_id','geometry']].to_file('./nodes_out.shp')

    #########################
    #can use a raster to add elevations to the nodes:
    ################
    #osmnx.elevation.add_node_elevations_raster(G, filepath, band=1, cpus=None)
    # merge the elevation data for the road from the raster file
    #add an elevation atribute to the road nodes that way when we deal with them later we are chilling 
    
    
                
    
    nodes_out84=nodes_out.to_crs('EPSG:4326')
    
    dtm = rxr.open_rasterio(dem_name, masked=True).squeeze() # mask no data values
    #dtm.rio.crs # to check the existing CRS
    dtm_m= dtm.rio.reproject('EPSG:4326')
    #dtm_r = rasterio.open('./USGS_13_n33w088_20220728.tif')
    #dtm_clipped = dtm.rio.clip(edges_out.geometry,
                                          # This is needed if your GDF is in a diff CRS than the raster data
    #                                      edges_out.crs)
               

# fig, ax = plt.subplots(figsize=(10, 10))

# # We plot with the zeros in the data so the CHM can be better represented visually
# ep.plot_bands(dtm,extent=plotting_extent(dtm,dtm.rio.transform()),  # Set spatial extent
#               cmap='Greys',
#               title="Uniontown, AL",
#               scale=False,
#               ax=ax)

# nodes_out.plot(ax=ax,
#                        marker='s',
#                        markersize=45,
#                        color='purple')
# ax.set_axis_off()
# plt.show()

    z_stats = rs.zonal_stats(nodes_out84,
                                        dtm_m.values,
                                        nodata=-999,
                                        affine=dtm_m.rio.transform(),
                                        geojson_out=True,
                                        copy_properties=True,
                                        stats="median")
    z_stats_df = gpd.GeoDataFrame.from_features(z_stats)
    nodes_out=nodes_out.merge(z_stats_df[['n_id','median']], on='n_id')
    nodes_out.columns=['n_id', 'x', 'y', 'geometry', 'elevation']
    
    demand_list = []
    for i in nodes_out['n_id']:
        if i in nodes.loc[nodes['n_demand'] > 0]['n_id']:
            demand_list.append(nodes.loc[nodes['n_id'] == i]['n_demand'])
        else:
            demand_list.append(0)
            
    nodes_out['n_demand'] = demand_list
        
    G3=nx.MultiDiGraph()
    
    #G2.add_nodes_from(N_list_df,x=N_list_df['x'],y=)
    
    for i in range(len(nodes_out)):
          G3.add_node(nodes_out['n_id'][i],x=nodes_out['x'][i],y=nodes_out['y'][i], 
                      elevation=nodes_out['elevation'][i])
    for i in range(len(edges_out)):
        G3.add_edge(edges_out['uN'][i],edges_out['vN'][i], weight=float(edges_out['length'][i]),
                    u=edges_out['u'][i],v=edges_out['v'][i])    
    
    G3 = G3.to_undirected()
#get the arcs
    f = open('clust_' + str(a+1) + '_road_arcs_utown.txt','w')
    for j, k, l in G3.edges:
        distance = G3.edges[j, k, l]['weight']
        f.write(str(j) + " " + str(k) + " " + str(distance) +'\n')
    f.close()
    
    nodes_out_li.append(nodes_out)
    edges_out_li.append(G3.edges)
    
nodes_df0 = nodes_out_li[0]
df_count = len(nodes_df0)
already_there = set(nodes_out_li[0]['n_id'])
for i in nodes_out_li[1:]:
    for j in range(0, len(i)):
        if i.iloc[j]['n_id'] not in already_there:
            nodes_df0.loc[df_count] = i.iloc[j]
            df_count+= 1
            already_there.add(i.iloc[j]['n_id'])
            
#!!!Make sure to output the cluster with the node


#for each house connected to a demand node 
# demand_dict = {}
# for i in range(0, len(build_shp.geometry)):
#     p = build_shp.geometry.iloc[i]
#     pot_pts = nearest_points(build_shp.geometry.iloc[i], MultiPoint(list(nodes_out.geometry)))
#     if len(pot_pts) == 1:
#         node_row = nodes_out.loc[nodes_out['geometry'] == pot_pts[0]]
#         if len(node_row) > 0:
#             if list(node_row['n_id'])[0] not in demand_dict:
#                 demand_dict[list(node_row['n_id'])[0]] = 1
#             else:
#                 demand_dict[list(node_row['n_id'])[0]] += 1
                
#     else:
#         node_row = nodes_out.loc[nodes_out['geometry'] == pot_pts[1]]        
#         if list(node_row['n_id'])[0] not in demand_dict:
#             demand_dict[list(node_row['n_id'])[0]] = 1
#         else:
#             demand_dict[list(node_row['n_id'])[0]] += 1
#go through each row and add the dictionary id to the corresponding dataframe
#0 if there is no houses that are connected to a given road node
# n_demand = []
# for i in nodes_out['n_id']:
#     if i not in demand_dict:
#         n_demand.append(0)
#     else:
#         n_demand.append(demand_dict[i])
        
# nodes_out['n_demand'] = n_demand

# term_nodes = []

# for i in range(len(nodes_out)):
#     name = nodes_out.iloc[i]['n_id']
#     if nodes_out.iloc[i]['n_demand'] > 0:
#         term_nodes.append(name)


# G2=nx.MultiDiGraph()

#G2.add_nodes_from(N_list_df,x=N_list_df['x'],y=)

# for i in range(len(nodes_out)):
#       G2.add_node(nodes_out['n_id'][i],x=nodes_out['x'][i],y=nodes_out['y'][i], 
#                   elevation=nodes_out['elevation'][i])
# for i in range(len(edges_out)):
#     G2.add_edge(edges_out['uN'][i],edges_out['vN'][i], weight=float(edges_out['length'][i]),
#                 u=edges_out['u'][i],v=edges_out['v'][i])    

# G3 = G2.to_undirected()
# #####pick it up from here to figure out why we have edges with lengths longer than 101 m
# test = [[float(nodes_out.loc[nodes_out['n_id'] == a]['x']), float(nodes_out.loc[nodes_out['n_id'] == a]['y'])] for a in term_nodes]
# test1 = distance_matrix(test, test)

#dist_df = pd.DataFrame(test1, columns = list(complete_df['n_id']), index = list(complete_df['n_id']))

# selected_data = test
#choose number of clusters with k
# ngroup = 200
# clustering_model = AgglomerativeClustering(n_clusters=ngroup, affinity='euclidean', linkage='ward')
# a = clustering_model.fit(selected_data)
# b = clustering_model.labels_

# n_clust = []

# for i in range(len(nodes_out)):
#     row = nodes_out.iloc[i]
#     name = row['n_id']
#     if name not in term_nodes:
#         n_clust.append(-1)
#     else:
#         clust_idx = term_nodes.index(name)
#         clust = b[clust_idx]
#         n_clust.append(clust)
    
#add lonlat back 
lat = []
lon = []

for i in nodes_df0['n_id']:
    row1 = nodes_df0.loc[nodes_df0['n_id'] == i]
    if len(row1) > 0:
        row1 = nodes_df0.loc[nodes_df0['n_id'] == i]
        inProj = Proj(init='epsg:2163')
        outProj = Proj(init='epsg:4326')
        lon1, lat1 = transform(inProj, outProj, row1['x'], row1['y'])
        lat.append(float(lat1))
        lon.append(float(lon1))
    else:
        lon1, lat1 = float(row1['x']), float(row1['y'])
        lat.append(lat1)
        lon.append(lon1)
    

nodes_df0['lat'] = lat
nodes_df0['lon'] = lon

# term_nodes_dict = dict()
# for i in range(0, ngroup):
#     temp_node_list = []
#     for j in range(len(term_nodes)):
#         clust_num = b[j]
#         if clust_num == i:
#             temp_node_list.append(term_nodes[j])
#     term_nodes_dict[i] = temp_node_list

#nx.algorithms.approximation.steinertree.metric_closure(G3.to_undirected())
# mst_list = []
# for i in range(0, ngroup):
#     mst = nx.algorithms.approximation.steinertree.steiner_tree(G3, term_nodes_dict[i], weight = 'weight')
#     mst_list.append(mst)

n_clust = []
for i in nodes_df0['n_id']:
    for j in range(len(mst_list)):
        if i in mst_list[j].nodes:
            n_clust.append(j)
            break
        if j == len(mst_list)-1:
            n_clust.append(-1)
nodes_df0['cluster'] = n_clust
# build_df2 = pd.DataFrame({'n_id': build_shp['n_id'],
#                           'x': build_shp['geometry'].x,
#                           'y': build_shp['geometry'].y,
#                           'geometry': build_shp['geometry'],
#                           'elevation': build_shp['V3']})

# complete_df = nodes_out.append(build_df2, len(nodes_out))


#for visualization
#plt.scatter(complete_df.x, complete_df.y, c=complete_df.cluster, alpha = 0.6, s=1)

multiline_list = []
m1_list = []
for i in range(0, ngroup):
    if len(edges_out_li[i]) == 0:
        continue
    mst_temp_edges = list(edges_out_li[i])
    multilinestring_list = []
    for q, r, s in mst_temp_edges:
        n1 = q
        n2 = r
        n1_row = nodes_df0.loc[nodes_df0['n_id'] == n1]
        n2_row = nodes_df0.loc[nodes_df0['n_id'] == n2]
        n1_pt = n1_row['geometry']
        n2_pt = n2_row['geometry']
        line = ((float(n1_pt.x), float(n1_pt.y)), (float(n2_pt.x), float(n2_pt.y)))
        line2 = ((float(n1_row['lon']), float(n1_row['lat'])), (float(n2_row['lon']), float(n2_row['lat'])))
        multilinestring_list.append(line2)
        m1_list.append(line2)
    multiline = MultiLineString(multilinestring_list)
    multiline_list.append(multiline)
    
    driver = ogr.ogr.GetDriverByName('Esri Shapefile')
    
    pipelayoutfile = str(ngroup) + '_cluster_' + str(i) + 'road_arcs' + '.shp'
    ds = driver.CreateDataSource(pipelayoutfile)
    layer = ds.CreateLayer('', None, ogr.ogr.wkbMultiLineString)
    # Add one attribute
    layer.CreateField(ogr.ogr.FieldDefn('id', ogr.ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    
    ## If there are multiple geometries, put the "for" loop here
    
    # Create a new feature (attribute and geometry)
    feat = ogr.ogr.Feature(defn)
    feat.SetField('id', 123)
    
    # Make a geometry, from Shapely object
    geom = ogr.ogr.CreateGeometryFromWkb(multiline.wkb)
    feat.SetGeometry(geom)
    
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
    
    # Save and close everything
    ds = layer = feat = geom = None
    
b = MultiLineString(m1_list)
b_dict = {'type': 'MultiLineString', 'coordinates': tuple(m1_list)}

# schema of the resulting shapefile
schema = {'geometry': 'MultiLineString','properties': {'id': 'int'}}
# save 
with fiona.open('multiline2.shp', 'w', driver='ESRI Shapefile', schema=schema)  as output:
     output.write({'geometry':b_dict,'properties': {'id':1}})


#csv for Dr. Schwetschenau
#for this df you are going to have to find a way to unpack nodes_out_li

wb = Workbook()
filename = "Road_Nodes_Utown_Demand"
s1 = wb.add_sheet('sheet1')
s1.write(0, 0, 'id')
s1.write(0, 1, 'lat')
s1.write(0, 2, 'lon')
s1.write(0, 3, 'elev')
s1.write(0, 4, 'cluster')
s1.write(0, 5, 'demand')
s1.write(0, 6, 'easting')
s1.write(0, 7, 'northing')
#record the information for all the roadpoints
counter = 1
for j in range(len(nodes_df0)):
    node_x = float(nodes_df0['lon'][j])
    node_y = float(nodes_df0['lat'][j])
    node_xft = float(nodes_df0['x'][j])
    node_yft = float(nodes_df0['y'][j])
    node_elev = float(nodes_df0['elevation'][j])
    node_cluster = int(nodes_df0['cluster'][j])
    node_id = str(nodes_df0['n_id'][j])
    node_demand = int(nodes_df0['n_demand'][j])
    
    s1.write(counter, 0, node_id)
    s1.write(counter, 1, node_y)
    s1.write(counter, 2, node_x)
    s1.write(counter, 3, node_elev)
    s1.write(counter, 4, node_cluster)
    s1.write(counter, 5, node_demand)
    s1.write(counter, 6, node_xft)
    s1.write(counter, 7, node_yft)
    #     if i == node_cluster:
    #         clust.write(counter, 0, str(node_id))
    #         clust.write(counter, 1, float(node_x))
    #         clust.write(counter, 2, float(node_y))
    #         clust.write(counter, 3, float(node_elev))
    #         clust.write(counter, 4, float(node_xft))
    #         clust.write(counter, 5, float(node_yft))
    #         clust.write(counter, 6, float(node_demand))
    #         counter += 1
    #         f.write(str(node_id) + " " + '%s %s %s %s %s %s\n'%(node_x, node_y, node_elev, node_xft, node_yft, node_demand))
    
    # f.close()
    counter += 1
            
wb.save(filename + '.csv')
#this is the real output file that the optimization model is going to use
nodes_out.to_csv('Uniontown_df.csv')
    
# clusterfilename = 'Centralized_elevcluster' + str(1) + '.csv'

# # load the files for building locations 
# clusterfile = os.path.realpath(os.path.join(os.path.dirname('IP_Decentralized'), '..')) + '\\' + clusterfilename
# building_coords = readClusterFile(clusterfile)
# building_coords = building_coords[:, :2]
# # load the building coordinate file
# building_coords = pd.read_csv(clusterfile, index_col=0)
# # convert to a shapefile to match nodes objective above - these will be merged into the graph
# build_shp = gpd.GeoDataFrame(building_coords, 
#             geometry=gpd.points_from_xy(building_coords.V1, building_coords.V2),
#             crs="EPSG:4326")

# build_shp=build_shp.to_crs("EPSG:2163")
# build_shp.insert(4,'n_id',None)
# counter_N=1
# for index, row in build_shp.iterrows():
#     build_shp.loc[index,'n_id']= 'B'+str(counter_N)
#     counter_N+=1
    
# #break each square up into a list where the order is top right, top left, bottom left, bottom right and find cnetroids
# long_len = xmax - xmin
# lat_len = ymax - ymin
# partition = 9
# lon_dif = long_len / partition
# lat_dif = lat_len / partition

# boundaries_list = []
# #divides the city into squares
# for i in range(partition):
#     for j in range(partition):
#         left_x = xmin + i*lon_dif
#         right_x = xmin + (i+1)*lon_dif
#         bottom_y = ymin + j*lat_dif
#         top_y = ymin + (j+1)*lat_dif
#         #output list with format left, top, right, bottom
#         boundaries_list.append((left_x, top_y, right_x, bottom_y))

# pot_treat = []
# for i in boundaries_list:
#     temp_dict = {}
#     xL = i[0]
#     xR = i[2]
#     yT = i[1]
#     yB = i[3]
#     for j in range(len(build_shp)):
#         build_lon = build_shp.V1[j+1]
#         build_lat = build_shp.V2[j+1]
#         if xL <= build_lon <= xR and yB <= build_lat <= yT:
#             temp_dict['B' + str(j + 1)] = build_shp.V3[j+1]
#     if len(temp_dict) <= 0:
#         pass
#     else:
#         min_val = min(temp_dict.values())
#         for k in temp_dict:
#             if temp_dict[k] == min_val:
#                 pot_treat.append(k)
#                 break
# pot_treat = list(set(pot_treat))
# ##loading dictionary so I don't have to waste time coding it again

# file1 = open(os.path.realpath(os.path.join(os.path.dirname('IP_Decentralized'), '..')) + "\\Columbia_all_building_road_paths_dict_seg.pkl", "rb")
# loaded_dict = pickle.load(file1)

# #load the graph so I don't have to waste time coding it again
# #Gtest = nx.read_gpickle("graph_dictionary_full.pkl")

# #connecting the building centroids to the road based on this github code:
# ######################################
# # There may be useful code in this function 
# #    https://github.com/ywnch/toolbox/blob/master/toolbox.py
# # possiblly helpful packages
# #import rtree #- helpful spatial indexing functions for some GIS type operationgs
# #import itertools  #
# build_dict_low = {}
# for i in pot_treat:
#     #need to go back to loaded_dict to fix the fact that build_road_dict['B#', 'B#self'][1] = ['B#', 'B#self'] instead of [('B#', 'B#self')]
#     #this would allow me to uncomment this bit of code
#     #we need to tuple up the path that way when we iterate we don't have the weird tuple issue I mentioned earlier
#     build_dict_low[i, i] = loaded_dict[i, i]
#     for ii in range(0,len(build_shp)):
#         ni=build_shp.iloc[ii].n_id
#         if ni != i:
#             build_dict_low[ni, i] = loaded_dict[ni, i]

# graph_dict_low = open(".\\Treat_building_road_paths_dict_new_seg.pkl", "wb")
# pickle.dump(build_dict_low, graph_dict_low)

# #go through the msts
# fileID = os.path.realpath(os.path.join(os.path.dirname('IP_Decentralized'), '..')) + '\\' + '5_mstroad_1.txt'
# file = np.genfromtxt(fileID, delimiter=" ")
# file = file[:,1:]
# file_id = np.genfromtxt(fileID, delimiter=" ", dtype = str)
# file_id = file_id[:,0]

# #fix this, right now it includes road nodes that aren't in file_id


# cluster_pot_treat = []
# for i in file_id:
#     if i in pot_treat:
#         cluster_pot_treat.append(i)

# mst1_dict = {}
# for i in file_id:
#     for j in cluster_pot_treat:
#         if i != j and i[0] == "B":
#             mst1_dict[i, j] = build_dict_low[i, j]

# mst_edges = set()
# mst_nodes = set()

# for i, j in mst1_dict:
#     for k in mst1_dict[i, j][1]:
#         mst_nodes.add(k[0])
#         mst_nodes.add(k[1])
#         mst_edges.add(k)
        
# arcs = mst_edges

#background plot:
# fig, ax = plt.subplots(1, figsize = (50, 50))

# #creating building points

# cluster_df=pd.DataFrame(columns=["Building","Latitude","Longitude","Elevation"],\
#                                 index=mst_nodes)
# inProj = Proj(init = 'epsg:2163')
# outProj = Proj(init='epsg:4326')


# for i in mst_nodes:
#     if i not in file_id:
#         continue
#     row = list(file_id).index(i)
#     lat, long = file[row,1], file[row,0]
#     elev = file[row, 2]

#     temp=[i,lat,long,elev]
#     cluster_df.loc[i]=temp
        
# clustergdf = gpd.GeoDataFrame(
#     cluster_df, geometry=gpd.points_from_xy(cluster_df.Longitude, cluster_df.Latitude))


# #creating the roadlines
# #reads sf files from R code in the spatial features tab
# #has to replace lower case C with capital C

# #
# clustermultilinelist = []
# for a in arcs:
#     i, j = a[0], a[1]
#     frompointlon, frompointlat = nodes_out.loc[nodes_out['n_id'] == i]['lon'], nodes_out.loc[nodes_out['n_id'] == i]['lat']
#     frompoint = Point(float(frompointlon), float(frompointlat))
#     topointlon, topointlat = nodes_out.loc[nodes_out['n_id'] == j]['lon'], nodes_out.loc[nodes_out['n_id'] == j]['lat']
#     topoint = Point(float(topointlon), float(topointlat))
#     line = LineString([frompoint, topoint])
#     clustermultilinelist.append(line)

# clustermultiline = MultiLineString(clustermultilinelist)
# driver = ogr.GetDriverByName('Esri Shapefile')

# pipelayoutfile = 'MSTRoad_' + 'U_town' + '_pipelayout'+ str(i) + '.shp'
# ds = driver.CreateDataSource(pipelayoutfile)
# layer = ds.CreateLayer('', None, ogr.wkbMultiLineString)
# # Add one attribute
# layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
# defn = layer.GetLayerDefn()

# ## If there are multiple geometries, put the "for" loop here

# # Create a new feature (attribute and geometry)
# feat = ogr.Feature(defn)
# feat.SetField('id', 123)

# # Make a geometry, from Shapely object
# geom = ogr.CreateGeometryFromWkb(clustermultiline.wkb)
# feat.SetGeometry(geom)

# layer.CreateFeature(feat)
# feat = geom = None  # destroy these

# # Save and close everything
# ds = layer = feat = geom = None

# pipelines = gpd.read_file(pipelayoutfile)
# pipelines.plot(ax = ax, color = 'black')

# #pipelines = gpd.read_file('C' + clustername[1:] + '.shp')
# #pipelines.plot(ax = ax, color = 'black')
# elevation_list = []
# for i in cluster_pot_treat:
#     idx = list(file_id).index(i)
#     lat, lon, elev = file[idx,:]
#     elevation_list.append(elev)
# min_elev = min(elevation_list)
# min_idx = elevation_list.index(min_elev)
# min_build = cluster_pot_treat[min_idx]

# clustergdf.plot(ax = ax, column = 'Elevation', legend = True)
# plt.scatter(file[:,0][list(file_id).index(min_build)], file[:,1][list(file_id).index(min_build)], facecolors = 'None', edgecolors = 'r')
# for lat, lon, label in zip(clustergdf.geometry.y, clustergdf.geometry.x, clustergdf.Building):
#     ax.annotate(label, xy=(lon, lat), xytext=(lon, lat))
# plt.show()

# graph_dict_low.close()      




