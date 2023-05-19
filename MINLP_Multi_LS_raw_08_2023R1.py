# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:55:13 2021

@author: yunus
"""
from osgeo import ogr
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from math import sin, cos, sqrt, atan2, radians
import sys
import geopandas as gpd
from shapely.geometry import Polygon, box, Point, LineString, MultiLineString
import xlwt
from xlwt import Workbook
import gurobipy as gp
from gurobipy import GRB
import os
# approximate radius of earth in km
xmin = -87.52739617137372
xmax = -87.47#841138757
ymin = 32.42075991564647
ymax = 32.469#87035515
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
    return distance

def readClusterFile(fileID):
    file = np.genfromtxt(fileID, delimiter=" ")
    if np.count_nonzero(file) <= 5:
        file_list = file
    else:
        file_list = file[:,1:]
    return file_list

def readClusterfileID(fileID):
    file_indexes = np.genfromtxt(fileID, delimiter=" ", dtype = str)
    if np.count_nonzero(file_indexes) <= 5:
        file_id = file_indexes[0]
    else:
        file_id = file_indexes[:,0]
    return file_id

def readArcs(fileID):
    file = np.genfromtxt(fileID, delimiter=" ", dtype = str)
    return file


#note: the slopes in this function take the from node as the start point(i) and the to node as the endpoint (j)
def findDistances(tree, dataframe):
    returnDictdist = {}
    for i, j in tree:
        lon1 = dataframe.loc[dataframe['n_id'] == i]['lon']
        lat1 = dataframe.loc[dataframe['n_id'] == i]['lat']
        lon2 = dataframe.loc[dataframe['n_id'] == j]['lon']
        lat2 = dataframe.loc[dataframe['n_id'] == j]['lat']
        distance = haversinedist(lat1, lon1, lat2, lon2)
        #meters divded meters
        returnDictdist[i,j] = distance * 1000
    
    return returnDictdist   

#this section of code makes sure the directed arcs drain into the source node
    
#this function takes a point and the list of arcs and the list of already accounted for arcs and finds
#all the arcs that are connected to a node, which have not already been insepcted in the repetition variable
#this section of code makes sure the directed arcs drain into the source node

#this function takes a point and the list of arcs and the list of already accounted for arcs and finds
#all the arcs that are connected to a node, which have not already been insepcted in the repetition variable
def findconnectingnodes(source, arclist, repetition):
    returnlist = list()
    for i in range(len(arclist)):
        for j in source:
            if (arcs[i][0] == j or arcs[i][1] == j) and arclist[i] not in repetition and arclist[i][::-1] not in repetition:
                returnlist.append(arclist[i])
    return returnlist

#this function goes through the directed mst and makes sure the directions go the right way
def correctFlow(arcs1, outlet):
    visitedToNodes = []
    returnArcList = []
    source_index = outlet
    visitedToNodes.append(source_index)
    tempNodes = [source_index]
    #travels through all the arcs of the mst switches flow direction if it is facing the wrong way
    while len(visitedToNodes) <= len(arcs):
        tempToList = findconnectingnodes(tempNodes, arcs1, returnArcList)
        for i in tempToList:
            if i[0] in visitedToNodes:
                rev_tup = i[::-1]
                returnArcList.append(rev_tup)
                visitedToNodes.append(rev_tup[0])
                tempNodes.append(rev_tup[0])
            else:
                returnArcList.append(i)
                visitedToNodes.append(i[0])
                tempNodes.append(i[0])
        tempNodes = tempNodes[-len(tempToList):]
        tempToList = []
        
    return returnArcList
#flips the arcs if 
def flip(arcList, out):
    returnArcs = []
    returnStarts = []
    for i in arcList:
        if i[0] == out:
            returnArcs.append(i[::-1])
            returnStarts.append(i[1])
        elif i[1] == out:
            returnArcs.append(i)
            returnStarts.append(i[0])
    return returnArcs, returnStarts

def correctFlow2(arcsR, outletR):
    todealwith = arcsR
    row_count = 0
    visited = set()
    returnArcs = np.array([0,0])
    toVisit = [outletR]
    while len(visited) < len(todealwith):
        toVisit2 = []
        for i in toVisit:
            for j in range(len(todealwith)):
                if i == todealwith[j, 0] and tuple(todealwith[j]) not in visited:
                    returnArcs = np.vstack((returnArcs, todealwith[j][::-1]))
                    toVisit2.append(todealwith[j,1])
                    visited.add(tuple(todealwith[j]))
                    row_count += 1
                elif i == todealwith[j, 1] and tuple(todealwith[j]) not in visited:
                    returnArcs = np.vstack((returnArcs, todealwith[j]))
                    visited.add(tuple(todealwith[j]))
                    toVisit2.append(todealwith[j,0])
                    row_count += 1
        toVisit = toVisit2
    return returnArcs[1:,:]
  


############################ initialize parameters
node_flow = 250 / (60 * 24) #250 gallons per day per household is converted into gallons/min
arb_min = 0.01
pipesize = [0.2, 0.25, 0.3,0.35,0.4, 0.45]
# fully installed costs    
pipesize_str, pipecost = gp.multidict({'0.05': 18, '0.06': 19, '0.08': 22, \
                                                   '0.1': 25, '0.15': 62,'0.2': 171, '0.25': 187, '0.3': 203, '0.35':230, '0.4': 246, '0.45':262})

# pipe costs without excavation
#pipesize_str, pipecost = gp.multidict({'0.05': 8.7, '0.06': 9.5, '0.08': 11, \
#                                       '0.1': 12.6, '0.15': 43.5,'0.2': 141, '0.25': 151, '0.3': 161})    
                
#fix pipe excavation trapezoid
#calculate the pirce per galon pumped
excavation = 0#25
bedding_cost_sq_ft = 0#6
capital_cost_pump_station = 171000
ps_flow_cost = 0.38
ps_OM_cost = 359317
treat_om = 237000
fixed_treatment_cost = 44000
added_post_proc = 8.52 #for gallons per day so use arcFlow values
collection_om = 209
hometreatment =0


#note: if you did this for the other models you don't need to do it for this one
#after you run this in the kernel then you start adding your sheets
wb = Workbook()
GravitywMultiPumpsSheet = wb.add_sheet("GravitywMultiPumps")
GravitywMultiPumpsSheet.write(0, 0, 'Cluster_Name')
GravitywMultiPumpsSheet.write(0, 1, 'Obj1')
GravitywMultiPumpsSheet.write(0, 2, 'Obj2')
GravitywMultiPumpsSheet.write(0, 3, 'Obj3')
GravitywMultiPumpsSheet.write(0, 4, 'Obj4')
GravitywMultiPumpsSheet.write(0, 5, 'Obj')
GravitywMultiPumpsSheet.write(0, 6, 'Objective + Additional Costs')

Pumps = wb.add_sheet("pumps_loc_cluster")
Pumps.write(0, 0, 'cluster')
Pumps.write(0, 1, 'Pump_Arc_Locations')
Pumps.write(0, 2, 'Pump_Arc_Mid_Lon')
Pumps.write(0, 3, 'Pump_Arc_Mid_Lat')
pumpcounter = 0

#define aquifer boundaries:
#we need to know aquifer boundaries to identify these potential treatment nodes
aquifers = gpd.read_file("C:\\Users\\yunus\\OneDrive\\Desktop\\Columbia_School_Work\\Alabama_Water_Project\\WW_FINAL\\us_aquifers.shx")
utown_poly = Polygon([[xmin, ymin], [xmin, ymax], [xmax,ymax], [xmax, ymin]])
aquifers_utown = gpd.clip(aquifers, utown_poly)

ngroup=10
for cluster in range(1, (ngroup+1)):
    arcsfilename = 'clust_' + str(cluster) + '_road_arcs_utown.txt'
    arcsfile = os.path.realpath(os.path.join(os.path.dirname('MST_Decentralized'))) + '\\' + arcsfilename
    arcsDist = readArcs(arcsfile)
    if len(arcsDist) == 0:
        GravitywMultiPumpsSheet.write(cluster, 1, 0)
        GravitywMultiPumpsSheet.write(cluster, 2, 0)
        GravitywMultiPumpsSheet.write(cluster, 3, 0)
        GravitywMultiPumpsSheet.write(cluster, 4, 0)
        GravitywMultiPumpsSheet.write(cluster, 5, hometreatment + added_post_proc + fixed_treatment_cost)
        continue
    arcs = arcsDist[:,:-1]
    
    road_nodes = set()
    demand_nodes = []
    
    arcDistances = dict()
    for a, b, c in arcsDist:
        road_nodes.add(a)
        road_nodes.add(b)
        arcDistances[a, b] = float(c)
        arcDistances[b, a] = float(c)
    
    df = pd.read_csv('Uniontown_df.csv')
    
    #determine which points are able to be used as treatment
    treatment = []
    for i in df['n_id']:
        row = df.loc[df['n_id'] == i]
        rowlat = float(row['lat'])
        rowlon = float(row['lon'])
        geo = Point([rowlon, rowlat])
        if aquifers_utown.contains(geo).all() and i in road_nodes and int(row['n_demand']) > 0:
            treatment.append(1)
        elif i in road_nodes:
            treatment.append(0.1)
        else:
            treatment.append(0)
    df['treatment'] = treatment
    #############################################################    
    #find the outlet node elevation using the dataframe
    #specify it can only come from a node with non zero demand
    #also within the aquifer
    #also have a case for when there the site is not above an aquifer
    if all(df['treatment'] != 0):
        min_elevation = min(list(df[df['treatment'] == 1]['elevation']))
    else:
        min_elevation = min(list(df[df['treatment'] == 0.1]['elevation']))
    outlet_node = df.loc[df['elevation'] == min_elevation]['n_id'].values[0]
    
    arcsnp = np.array([0, 0])
    for i in arcs:
        arcsnp = np.vstack((arcsnp, i))
    arcsnp = arcsnp[1:,:]

    arcs = correctFlow2(arcsnp, outlet_node)
    
    #connectivity list with corresponding slopes
    #the end node or treatment plant is the final coordinate which is at index 79 or the 80th coordinate
    #arcDistances = findDistances(arcs, df)   
    nodes = list()
    nodes_notup = list()
    arcFlow = dict()
    
    # for i in range(len(arcs)+1):
    #     tup = (i,)
    #     nodes.append(tup)
    #     nodes_notup.append(i)
    
    #pumpcap = dict()
    #arcs, arcSlopes = gp.multidict(arcSlopes)

            
    arcarray = np.array(arcs)
    fromcol = list(arcarray[:,0])
    tocol = list(arcarray[:, 1])
    for i in tocol:
        if i not in fromcol:
            end = i
    #fix this to reflect rainfall
    startpoints = [i for i in road_nodes if i[0] not in tocol]
    visitedpoints = []
    previous = []
    beforemergeval = 0
    for i in startpoints:
        currentpoint = i
        move = []
        while currentpoint != end:
            rowindex = int(np.where(arcarray[:,0] == currentpoint)[0])
            to = arcarray[rowindex,:][1]
            if (currentpoint, to) not in arcFlow:
                arcFlow[currentpoint, to] = sum(move) + float(df.loc[df['n_id'] == currentpoint]['n_demand'])*250/(60*24) #from 250 gallons per household per day
            else:
                #if we are merginnig we should give it all our water
                if previous not in visitedpoints[:-1]:
                    beforemergeval = arcFlow[tuple(previous)]
                    arcFlow[currentpoint, to] += beforemergeval
                else:
                    arcFlow[currentpoint, to] += beforemergeval
                
            previous = [currentpoint, to]
            visitedpoints.append(previous)
            currentpoint = to
            move.append(float(df.loc[df['n_id'] == currentpoint]['n_demand'])*250/(60*24))
    arcs = list(arcs)
    #Break up treatment plant node into a dummy node and a treatment plant node only a meter or two away.
    #sets it up as 100 meters away cause the elevation change might be a lot so this accomodates for that
    road_nodes.add(outlet_node + 'f')
    arcs.append(np.array([outlet_node, outlet_node + 'f'], dtype='<U11'))
    #nodes_notup.append((len(nodes)-1),)
    endnodelon = float(df.loc[df['n_id'] == outlet_node]['lon']) + 0.0001
    endnodelat = float(df.loc[df['n_id'] == outlet_node]['lat']) + 0.0001
    endnodeelev = float(df.loc[df['n_id'] == outlet_node]['elevation'])
    df2 = {'n_id': str(outlet_node) + 'f', 'x': 0, 'y': 0, 'geometry': Point(0,0), 'elevation': endnodeelev, 'n_demand': 0, 'lat': endnodelat, 'lon': endnodelon, 'cluster': -1, 'treatment': 1}
    df = df.append(df2, ignore_index = True)
    arcDistances[(outlet_node, outlet_node + 'f')] = 35
    
    endlinks = [(i, j) for i,j in arcFlow if j == outlet_node]
    for i, j in endlinks:
        if (outlet_node, outlet_node + 'f') in arcFlow:
            arcFlow[(outlet_node, outlet_node + 'f')] += arcFlow[i,j]
        else:
            arcFlow[(outlet_node, outlet_node + 'f')] = arcFlow[i,j]
    #nodes = twoDcluster
    
    # groundelev_dict = dict()
    
    # for i in road_nodes:
    #     index = int(i[0])
    #     if i[0] == len(threeDcluster):
    #     groundelev_dict[i] = threeDcluster[outlet_node, 2]
    #     else:
    #         groundelev_dict[i] = threeDcluster[index, 2]
        
    inflow = dict()
    inflow_count = 0
    #have to add a final node with a demand of zero for the outlet
    #demands = np.append(demands, 0)
    nodes2 = []
    for i in road_nodes:
        n_name = str(i)
        nodes2.append(n_name)
    
    for i in nodes2:
        if i == outlet_node + 'f':
            inflow[i] = -arcFlow[outlet_node, outlet_node + 'f']
        else:
            inflow[i] = float(df.loc[df['n_id'] == i]['n_demand'])*250/(60*24)
        
        
    building_num = 0
    for i in nodes2:
        df_row = df.loc[df['n_id'] == i]
        building_num += int(df_row['n_demand'])
    #this section is for the optimization of the model
    
###################################### opt model    
    #this section is for the optimization of the model
    
    m = gp.Model('Cluster1')
    #for multiple solutions
    
    #pipe diameter
    pipeflow = arcFlow.copy()
    for i, j in arcs:
        pipeflow[i,j] = pipeflow[i,j] / 15850
    #m.setParam('NonConvex',2) # allows non convex quadratic constraints
    m.Params.timeLimit = 1200
    #always run feasibilitytol and intfeastotal together
    #m.Params.feasibilitytol = 0.01
    #m.Params.optimalitytol = 0.01
    #m.Params.IntFeasTol = 0.1
    #m.Params.tunecleanup = 1.0
    #pipe diameter


    arc_sizes = m.addVars(pipeflow.keys(), pipesize, vtype = GRB.BINARY, name = "DIAMETER")
    pl = m.addVars(pipeflow.keys(), vtype = GRB.BINARY, name = "PUMPS")
    pc = m.addVars(pipeflow.keys(), lb = 0, vtype = GRB.CONTINUOUS, name = 'Pump Capacity')
    #pump location
    #pump capacity

    #node elevation excavation in meters
    #upper bound is arbritrary maximum depth assuming 1 foot or 0.3048 meters of cover beneath the surface is needed for the pipes
    #a lower bound variable is created but not used. In future models might need to implement that depending on the site (digging too deep for excavation is not feasible for many projects)
    elevation_ub = dict()
    elevation_lb = dict()
    for i in nodes2:
        elevation_ub[i] = float(df.loc[df['n_id'] == i]['elevation']) - 0.3048
        elevation_lb[i] = float(df.loc[df['n_id'] == i]['elevation']) - 30

    
    
    eIn = m.addVars(nodes2, lb = elevation_lb, ub = elevation_ub, name = 'In Node Elevation')
    eOut = m.addVars(nodes2, lb = elevation_lb, ub = elevation_ub, name = 'Out Node Elevation')
    #eIn[((len(nodes)-1),)] = elevation_ub[((len(threeDcluster)-2),)]
    eOut[outlet_node] = float(df.loc[df['n_id'] == outlet_node]['elevation'])
    nodeElevDif = m.addVars(nodes2, lb = 0, name = 'Difference Between Node Elevations')
    
 
    
    #select one type of pipe
    m.addConstrs((gp.quicksum(arc_sizes[i,j,k] for k in pipesize) == 1 for i,j in pipeflow.keys()), "single size chosen")
     
    m.addConstrs((
        nodeElevDif[i] == eOut[i] - eIn[i] for i in nodes2), 'In_Node_Out_Node_Difference')
            
    #if dealing with the whole system would do I but J makes more sense in terms of the lift station at the end
    # m.addConstrs(
    #     (slope[i,j] == (eIn[(i,)] - eIn[(j,)] + nodeElevDif[(i,)]) / arcDistances[i,j] for i,j in arcs), "slope")
    for i,j in pipeflow.keys():
        if j == outlet_node:
            #m.addConstr((0.001 <= (-eIn[(i,)] + eIn[(j,)] - nodeElevDif[(i,)]) / arcDistances[i,j]), "slope min" + str([i,j]))
            #m.addConstr((0.1 >= (-eIn[(i,)] + eIn[(j,)] - nodeElevDif[(i,)]) / arcDistances[i,j]), "slope max" + str([i,j]))
            pass
        else:
            m.addConstr(
                    (0.01 <= (eIn[i] - eIn[j] + nodeElevDif[i]) / arcDistances[i,j]), "slope min" + str([i,j]))
            m.addConstr(
                    (0.1 >= (eIn[i] - eIn[j] + nodeElevDif[i]) / arcDistances[i,j]), "slope max" + str([i,j]))
            
    m.addConstrs((
        arcDistances[i,j]*(gp.quicksum(arc_sizes[i, j, k] / k**(16/3) for k in pipesize)) * (pipeflow[i,j] / (11.9879))**2 <= eIn[i] - eIn[j] + nodeElevDif[i] for i,j in pipeflow.keys()), "Manning Equation")
    
    # for i in range(len(arcs)):
    #     if i == 0:
    #         pass
    #     elif arcs[i-1][0] not in arcarray[:,1]:
    #         pass
    #     else:
    #         m.addConstr((
    #             gp.quicksum(arc_size[]) <= d[arcs[i]]), "Ensures that the next pipe has to be bigger or same size")
    
    #assume flow area is half of the pipe area (embedded into equation)
    m.addConstrs((
        pipeflow[i,j] <= ((3.14/8)*gp.quicksum(arc_sizes[i,j,k]*k**2 for k in pipesize)) * 3 for i,j in pipeflow.keys()), "Velocity Max Constr")
    #m.addConstrs((
    #    pipeflow[i,j] >= ((3.14/8)*gp.quicksum(arc_sizes[i,j,k]*k**2 for k in pipesize)) * 0.6 for i,j in arcs), "Velocity Min Constr")
            
    
    for i,j in pipeflow.keys():
        m.addConstr((pl[i,j] == 1) >> (nodeElevDif[j] >= 0.0000000000000000000000000000000000001))
        m.addConstr((pl[i,j] == 0) >> (nodeElevDif[j] <= 0))
    
    m.addConstrs((
        pipeflow[i, j]*pl[i,j] <= pc[i,j] for i,j in pipeflow.keys()), "Pump Capacity Constraint")
    

    #width of trench is at least 3 ft or approximately 1 meter plus diameter
    #volume will be in cubic meters
    #cost is for excavation and infilling combined
    #for gravity you need pipe bedding (gravel underneath pipe so ground doesn't settle and pipe doesn't break)
    #4 in of bedding under gravity lie
    #$6 per square meter
    #bedding is first part excavation/infilling is second
    #cost accounts for 4 inches deep so just multiple by $6
    #discrete variables for pipe size so costs will be dictionary
    #obj1 = gp.quicksum(excavation * (1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * arcDistances[i,j] * 0.5 * ((elevation_ub[(i,)] - eIn[(i,)]) + (elevation_ub[(j,)] - eOut[(j,)])) for i, j in arcs)
    obj1 = gp.quicksum((1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * arcDistances[i,j] * bedding_cost_sq_ft +\
                       excavation * (1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * arcDistances[i,j] * 0.5 * \
                           ((elevation_ub[i] - eIn[i]) + (elevation_ub[j] - eOut[j])) for i, j in pipeflow.keys())

    obj2 = gp.quicksum(pl[i,j] * capital_cost_pump_station for i,j in pipeflow.keys()) 
    obj3 = gp.quicksum(arcDistances[i,j] * gp.quicksum(pipecost[str(k)] * arc_sizes[i, j, k] for k in pipesize) for i,j in pipeflow.keys())
    #pump operating costs yearly
    obj4 = collection_om * (building_num +1) + gp.quicksum(ps_OM_cost*pl[i,j] for i,j in pipeflow.keys()) + treat_om
    obj = obj1 + obj2 + obj3 + obj4 + fixed_treatment_cost + added_post_proc*arcFlow[outlet_node, outlet_node + 'f']*60*24 + hometreatment * (building_num)
    #obj = obj1 + obj2 + obj3
    
    #m.Params.Presolve = 0
    #m.Params.Method = 2
    #m.Params.PreQLinearize = 1
    #m.Params.Heuristics = 0.001
    
    m.setObjective(obj, GRB.MINIMIZE)
    #m.Params.tunetimelimit = 3600
    #m.tune()
    m.optimize()
    #p = m.presolve()
    #p.printStats()
    
    #p = m.presolve()
    #p.printStats()
    
    
    
    
    # print('The model is infeasible; computing IIS')
    # m.computeIIS()
    # if m.IISMinimal:
    #     print('IIS is minimal\n')
    # else:
    #     print('IIS is not minimal\n')
    # print('\nThe following constraint(s) cannot be satisfied:')
    # for c in m.getConstrs():
    #     if c.IISConstr:
    #         print('%s' % c.constrName)
    
    # GravitywMultiPumpsSheet.write(clusternumber, 1, obj1.getValue())
    GravitywMultiPumpsSheet.write(cluster, 1, obj1.getValue())
    GravitywMultiPumpsSheet.write(cluster, 2, obj2.getValue())
    GravitywMultiPumpsSheet.write(cluster, 3, obj3.getValue())
    GravitywMultiPumpsSheet.write(cluster, 4, obj4.getValue())
    GravitywMultiPumpsSheet.write(cluster, 5, m.objVal)
    GravitywMultiPumpsSheet.write(cluster, 6, m.objVal)   
    
    # if m.status == GRB.INFEASIBLE:
    #     relaxvars = []
    #     relaxconstr = []
    #     for i in m.getVars():
    #         if 'velocity[' in str(i):
    #             relaxvars.append(i)
                    
    #     for j in m.getConstrs():
    #         if 'slope[' in str(j):
    #             relaxconstr.append(j)
                    
    #     lbpen = [3.0]*len(relaxvars)
    #     ubpen = [3.0]*len(relaxvars)
    #     rhspen = [1.0]*len(relaxconstr)
                    
    #     m.feasRelax(2, False, relaxvars, lbpen, ubpen, relaxconstr, rhspen)
    #     m.optimize()
    
    # modelname = "Decentralized_Uniontown" + tot_clust_str + clustername + "gravityMultipump" + ".txt"
    # #m.write(modelname)
    # modelfile = open(modelname, "w")
    # modelfile.write('Solution Value: %g \n' % m.objVal)
    # for v in m.getVars():
    #     modelfile.write('%s %g \n' % (v.varName, v.x))
    # modelfile.close()
    
    #m.close()
  
    
    #pumpcounter=0
    for a,b in pl:
        a_lon = float(df.loc[df['n_id'] == a]['lon'])
        a_lat = float(df.loc[df['n_id'] == a]['lat'])
        b_lon = float(df.loc[df['n_id'] == b]['lon'])
        b_lat = float(df.loc[df['n_id'] == b]['lat'])
        if pl[a, b].X == 1:
            pumpcounter += 1
            Pumps.write(pumpcounter, 0, cluster)
            Pumps.write(pumpcounter, 1, str([a, b]))
            Pumps.write(pumpcounter, 2, (a_lon + b_lon)*0.5)
            Pumps.write(pumpcounter, 3, (a_lat + b_lat)*0.5)
    #mapping the boundaries of the system
    #how to plot a shapley polygon
    # uniontown = gpd.read_file('Uniontown_City_Limits.shp')
    # pointlist = list(uniontown.geometry)
    # polygon_feature = Polygon([[poly.x, poly.y] for poly in pointlist])
    # x, y = polygon_feature.exterior.xy
    #list of node elevations:
    # final_elevations = []
    # for i in eIn:
    #     if isinstance(eIn[i], float):
    #         value = eIn[i] + nodeElevDif[i].x
    #     else:
    #         value = eIn[i].x + nodeElevDif[i].x
    #     final_elevations.append(value)
    
    # for i in threeDcluster:
    #     final_elevations.append(i[2])
    
    
    #background plot:
    fig, ax = plt.subplots(1, figsize = (50, 50))
    
    #creating building points
    cluster_df=pd.DataFrame(columns=["Building","Latitude","Longitude","Elevation"],\
                                    index=nodes)

    for i in nodes2:
        elev=0
        lat=0
        long=0
        if type(nodeElevDif[i]) == int:
            elev=eIn[i].x + nodeElevDif[i]
        else:
            elev=eIn[i].x + nodeElevDif[i].x

        lat=float(df.loc[df['n_id'] == i]['lat'])
        long=float(df.loc[df['n_id'] == i]['lon'])

        temp=[i,lat,long,elev]
        cluster_df.loc[i]=temp
            
    clustergdf = gpd.GeoDataFrame(
        cluster_df, geometry=gpd.points_from_xy(cluster_df.Longitude, cluster_df.Latitude))
    #cluster_dict = {}
    pump_dict = {}
    #cluster_dict["Building"] = nodes_notup#[:-1]
    #cluster_dict["Latitude"] = list(threeDcluster[:,1])#[:-1]
    #cluster_dict["Longitude"] = list(threeDcluster[:,0])#[:-1]
    #cluster_dict["Elevation"] = final_elevations#[:-1]
    
    
    #pump locations
    pumpLocationsLon = []
    pumpLocationsLat = []
    pumpNames = []
    counter = 0
    for i,j in pl:
        i_lon = float(df.loc[df['n_id'] == i]['lon'])
        i_lat = float(df.loc[df['n_id'] == i]['lat'])
        j_lon = float(df.loc[df['n_id'] == j]['lon'])
        j_lat = float(df.loc[df['n_id'] == j]['lat'])
        for k in pc:
            if pl[i,j].X >= 0.9:
                loc_lon = (i_lon + j_lon) * 0.5
                loc_lat = (i_lat + j_lat) * 0.5
                pumpLocationsLon.append(loc_lon)
                pumpLocationsLat.append(loc_lat)
                pumpNames.append(counter)
                counter += 1
    
    
    pump_dict["Pump"] = pumpNames
    pump_dict["Latitude"] = pumpLocationsLat
    pump_dict["Longitude"] = pumpLocationsLon
    

    
    clusterpump_df = pd.DataFrame(pump_dict)
    clusterpumpgdf = gpd.GeoDataFrame(
        clusterpump_df, geometry = gpd.points_from_xy(clusterpump_df.Longitude, clusterpump_df.Latitude))
    # source_df = pd.DataFrame(source_dict)
    # sourcegdf = gpd.GeoDataFrame(
    #     cluster_df, geometry=gpd.points_from_xy(cluster_df.Longitude, cluster_df.Latitude))
    
    #creating the roadlines
    #reads sf files from R code in the spatial features tab
    #has to replace lower case C with capital C
    clustermultilinelist = []
    for i,j in pipeflow.keys():
        i_lon = float(df.loc[df['n_id'] == i]['lon'])
        i_lat = float(df.loc[df['n_id'] == i]['lat'])
        j_lon = float(df.loc[df['n_id'] == j]['lon'])
        j_lat = float(df.loc[df['n_id'] == j]['lat'])
        frompointlon, frompointlat = i_lon, i_lat
        frompoint = Point(frompointlon, frompointlat)
        topointlon, topointlat = j_lon, j_lat
        topoint = Point(topointlon, topointlat)
        line = LineString([frompoint, topoint])
        clustermultilinelist.append(line)
    
    clustermultiline = MultiLineString(clustermultilinelist)
    driver = ogr.GetDriverByName('Esri Shapefile')
    
    pipelayoutfile = "MST_STE_LS" + str(cluster) + 'pipelayout' + '.shp'
    ds = driver.CreateDataSource(pipelayoutfile)
    layer = ds.CreateLayer('', None, ogr.wkbMultiLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    
    ## If there are multiple geometries, put the "for" loop here
    
    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)
    
    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(clustermultiline.wkb)
    feat.SetGeometry(geom)
    
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
    
    # Save and close everything
    ds = layer = feat = geom = None
    
    pipelines = gpd.read_file(pipelayoutfile)
    pipelines.plot(ax = ax, color = 'black')
    
    #pipelines = gpd.read_file('C' + clustername[1:] + '.shp')
    #pipelines.plot(ax = ax, color = 'black')
    
    clustergdf.plot(ax = ax, column = 'Elevation', legend = True)
    clusterpumpgdf.plot(ax = ax, color = "red", marker = '^')
    plt.scatter(df.loc[df['n_id'] == outlet_node]['lon'], df.loc[df['n_id'] == outlet_node]['lat'], facecolors = 'None', edgecolors = 'r')
    for lat, lon, label in zip(clustergdf.geometry.y, clustergdf.geometry.x, clustergdf.Building):
        ax.annotate(label, xy=(lon, lat), xytext=(lon, lat))
    plt.show()
    

wb.save('v2_MINLP_raw_LS.xls')