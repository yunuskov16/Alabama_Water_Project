"""
Created on Mon Jan 23 5:46:00 pm 2023

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
    

# run this in the kernel to make sure you get the workbook started wb = Workbook()
#note: if you did this for the other models you don't need to do it for this one
#after you run this in the kernel then you start adding your sheets
################## initialize parameters
pipesize = [0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,0.45]
arb_min = 0.01
arb_max = 0.10
pipesize_str, pipecost = gp.multidict({'0.05': 8.7, '0.06': 9.5, '0.08': 11, \
                                       '0.1': 12.6, '0.15': 43.5,'0.2': 141, '0.25': 151, '0.3': 161, 
                                       '0.35':180, '0.4':190, '0.45':200})
    
excavation = 25
bedding_cost_sq_ft = 6
capital_cost_pump_station = 171000
ps_flow_cost = 0.38
ps_OM_cost = 175950
treat_om = 237000
hometreatment = 2000
fixed_treatment_cost = 18000
added_post_proc = 26.00 #for gallons per day so use arcFlow values
collection_om = 209

wb = Workbook()
STEPgravityDrainageSheet = wb.add_sheet("STEP_Gravity_Drainage")
STEPgravityDrainageSheet.write(0, 0, 'Cluster_Name')
STEPgravityDrainageSheet.write(0, 1, 'Obj1')
STEPgravityDrainageSheet.write(0, 2, 'Obj2')
STEPgravityDrainageSheet.write(0, 3, 'Obj3')
STEPgravityDrainageSheet.write(0, 4, 'Obj4')
STEPgravityDrainageSheet.write(0, 5, 'Obj')
STEPgravityDrainageSheet.write(0, 6, 'Objective + Additional Costs')

ngroups= 5
#define aquifer boundaries:
#we need to know aquifer boundaries to identify these potential treatment nodes
aquifers = gpd.read_file("C:\\Users\\yunus\\OneDrive\\Desktop\\Columbia_School_Work\\Alabama_Water_Project\\WW_FINAL\\us_aquifers.shx")
utown_poly = Polygon([[xmin, ymin], [xmin, ymax], [xmax,ymax], [xmax, ymin]])
aquifers_utown = gpd.clip(aquifers, utown_poly)

for cluster in range(1, (ngroups+1)):
    arcsfilename = 'clust_' + str(cluster) + '_road_arcs_utown.txt'
    arcsfile = os.path.realpath(os.path.join(os.path.dirname('MST_Decentralized'))) + '\\' + arcsfilename
    arcsDist = readArcs(arcsfile)
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
        
    #get rid of all the nodes that do not contribute flow
    # nodes2 = list(road_nodes.copy())
    # for i in road_nodes:
    #     if inflow[str(i)] == 0:
    #         nodes2.remove(i)
        
    #this section is for the optimization of the model
    #sets up the model
    #this model will be an MILP problem
    m = gp.Model('Cluster1')
    
    #the following are the settings I use to make the code run faster, look at gurobi api to learn what these things all do
    #basically I have commented out everything that I found useful, but no longer need
    #you might have to fiddle around with the time limit if you are dealing with big clusters 100+, as the code will given a lot more constraints to solve
    m.Params.timeLimit = 1200
    #always run feasibilitytol and intfeastotal together
    #m.Params.feasibilitytol = 0.01
    #m.Params.optimalitytol = 0.01
    #m.Params.IntFeasTol = 0.1
    #m.Params.tunecleanup = 1.0
    #m.Params.tunetimelimit = 3600
    #m.tune()
    #m.Params.Presolve = 0
    #m.Params.Method = 2
    #m.Params.PreQLinearize = 1
    #m.Params.Heuristics = 0.001
    #converting gallons per min into m^3 per second and setting up pumps, note pump capacity (pc) needs to be able to handle all system flow
    #pump location pumps are not used in this model and therefore not defined.
    pl = {}
    #pump capacity
    pc = {}
    pipeflow = arcFlow.copy()
    for i, j in arcFlow:
        pipeflow[i,j] = pipeflow[i,j] / 15850
        if (outlet_node, outlet_node + 'f') == (i,j):
            pl[i,j] = 1
            pc[i,j] = pipeflow[i,j]
        else:
            pl[i,j] = 0
            pc[i,j] = 0

    arc_sizes = m.addVars(pipeflow.keys(), pipesize, vtype = GRB.BINARY, name = "DIAMETER")

    #pipe diameter
    #in centimeters
    
    #node elevation excavation in meters
    #upper bound is arbritrary maximum depth assuming 1 foot or 0.3048 meters of cover beneath the surface is needed for the pipes
    #a lower bound variable is created because future models might need to implement that depending on the site (digging too deep for excavation is not feasible for many projects)
    elevation_ub = dict()
    
    for i in nodes2:
        elevation_ub[i] = float(df.loc[df['n_id'] == i]['elevation']) - 0.3048
        #elevation_lb[i] = threeDcluster[i[0], 2] - 5.3048
        
    #when installing pipes of different diameters we are going to use binary variables to indicate which one has been installed for what arc
    #each variable corresponds to a different pipe size (1 being the smallest and 11 being the largest)
    
    
    #piping cost is going to be a variable that adds up the cost for all the different pipes list previously
    
    #pipeflow is just arcflow but later you'll see we need to change it into m^3/s of water instead of gallons/min
    
    
    
    #because we are using gravity flow for this model for all the arcs
    #the lift stations need to be fashioned in a way where the wastewater is being pushed up to a level where it can continue
    #using gravity low to drain to the next nodes after its been pushed up, to do this we created two elevations for each node
    #eIn is the elevation for a pipe comming into a node, eOut is the elevation of a pipe comming out of a node
    #eOut is always higher or the same height as eIn. If a pump station occurs before a given node the eOut elevation > eIn elevation    
    eIn = m.addVars(nodes2, lb = -GRB.INFINITY, ub = elevation_ub, name = 'In Node Elevation')
    eOut = m.addVars(nodes2, lb = -GRB.INFINITY, ub = elevation_ub, name = 'Out Node Elevation')
    
    #for this model we already know the treatment plant is at ground level and has been split up between a dummy node and actual treatment 
    #plant node (the final node) because this model only uses gravity flow though what we are going to do is assume the treatment plant node
    #is around a foot in the ground and the dummy variable node is at ground level, this allows water to flow using gravity from
    #the dummy node to the treatment plant node
    eOut[outlet_node] = float(df.loc[df['n_id'] == outlet_node]['elevation'])
    
    #this will track the difference between eIn and eOut for every node
    nodeElevDif = m.addVars(nodes2, lb = 0, name = 'Difference Between Node Elevations')
    
    
    #this basically ensures that there will be no lift station except for the dummy node because this model uses gravity flow only
    #(with the exception being for the treatment plant node at ground level)
    for i in nodes2:
        if i != outlet_node:
            nodeElevDif[i] = 0
    #this ensures that the treatment plant node does not experience any lift station either
    #nodeElevDif[(len(nodes)-1,)] = 0 # sets the elevation difference in the last two (treatment) links to 0. 
        
    #applying lower and upper bounds to slope (to increase the speed of this you can define lb and ub when slope is being initialized)
    #which would decrease runtime but I prefer this way so I can relax the slopes of different arcs and see how the result is affected
    
    
            
    #select one type of pipe for every arc/link
    m.addConstrs((gp.quicksum(arc_sizes[i,j,k] for k in pipesize) == 1 for i,j in pipeflow.keys()), "single size chosen")
    #in gurobi you cannot multiple three variables or set variables to powers so you have to create general constraints and auxiliary variables
    #to do that stuff for you
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
        pipeflow[i,j] <= ((3.14/8)*gp.quicksum(arc_sizes[i,j,k]*k**2 for k in pipesize)) * 3 for i,j in pipeflow.keys()), "Velocity Max Constr")
    
    m.addConstrs((
        nodeElevDif[i] == eOut[i] - eIn[i] for i in nodes2), 'In_Node_Out_Node_Difference')
                
    #slope goes from i to j (from node to to node), this means a positive slope value means the pipe is going to a lower elevation
    #ensuring for gravity flow (note: this would not be the slope value in real life this only applies to this model)
    m.addConstrs((
        arcDistances[i,j]*(gp.quicksum(arc_sizes[i, j, k] / k**(16/3) for k in pipesize)) * (pipeflow[i,j] / (11.9879))**2 <= eIn[i] - eIn[j] + nodeElevDif[i] for i,j in pipeflow.keys()), "Manning Equation")
    

    #width of trench is at least 3 ft or approximately 1 meter plus diameter (see objective 1)
    #volume will be in cubic meters
    #cost is for excavation and bedding added together
    #for gravity you need pipe bedding (gravel underneath pipe so ground doesn't settle and pipe doesn't break)
    #4 in of bedding under gravity is incorporated into the beddding cost per square foot
    #$6 per square meter
    #bedding is first part excavation/infilling is second
    #cost accounts for 4 inches deep so just multiple by $6
    
    inindex, outindex = outlet_node, outlet_node + 'f'

    #excavation/infilling/bedding cost
    obj1 = gp.quicksum((1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * arcDistances[i,j] * bedding_cost_sq_ft +\
                       excavation * (1 + gp.quicksum(arc_sizes[i,j,k]*k for k in pipesize)*0.01) * arcDistances[i,j] * 0.5 *\
                           ((elevation_ub[i] - eIn[i]) + (elevation_ub[j] - eOut[j])) for i, j in pipeflow.keys())
    #pump installation costs
    obj2 = gp.quicksum(pl[i,j] * capital_cost_pump_station for i,j in pipeflow.keys()) #+ pc[inindex,outindex] * ps_flow_cost
    #piping capital costs
    obj3 = gp.quicksum(arcDistances[i,j] * gp.quicksum(pipecost[str(k)] * arc_sizes[i, j, k] for k in pipesize) for i,j in pipeflow.keys()) 
    #pump operating costs yearly
    obj4 = collection_om * (building_num+1) + ps_OM_cost + treat_om#there is only one OM cost because only one pump station in this model
    obj = obj1 + obj2 + obj3 + obj4 + fixed_treatment_cost + ((arcFlow[inindex,outindex])) * added_post_proc * 60 * 24 + hometreatment * (building_num)  #converts gals/min into gals per day
    #obj = obj1 + obj2 + obj3
    
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()  
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
    
    #rewrite this and call it the capital cost pump house
    
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
    STEPgravityDrainageSheet.write(cluster, 1, obj1.getValue())
    STEPgravityDrainageSheet.write(cluster, 2, obj2.getValue())
    STEPgravityDrainageSheet.write(cluster, 3, obj3.getValue())
    STEPgravityDrainageSheet.write(cluster, 4, obj4)
    STEPgravityDrainageSheet.write(cluster, 5, m.objVal)
    
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
    # modelname = "Decentralized_Uniontown_" + tot_clust_str + '_' + clustername + "STEP_Gravitydrainage" + ".txt"
    # #m.write(modelname)
    # modelfile = open(modelname, "w")
    # modelfile.write('Solution Value: %g \n' % m.objVal)
    # for v in m.getVars():
    #     modelfile.write('%s %g \n' % (v.varName, v.x))
    # modelfile.close()
    
    #Get the other solutions that are sub-optimal:
    # var_names = {}
    # for i in m.getVars():
    #     var_names[i.varName] = ''
    
    # for i in range(m.SolCount):
    #     m.Params.SolutionNumber = i
    #     modelname = str(i) + "st optimal solution " + clustername + "DecentralizedSTEPGravityDrainage" + ".txt"
    #     #m.write(modelname)
    #     modelfile = open(modelname, "w")
    #     modelfile.write('Solution Value: %g \n' % m.PoolObjVal)
    #     #make list of tuples:
    #     counter = 0
    #     sub_var = m.Xn
    #     for a in var_names:
    #         var_names[a] = sub_var[counter]
    #         counter += 1

    #     for j, k in var_names.items():
    #         modelfile.write('%s %g \n' % (j, k))
    #     modelfile.close()
    #m.close()
    
    #mapping the boundaries of the system

    
    
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
    
    pipelayoutfile = 'MST_STE_grav' + str(cluster) + '_pipelayout' + '.shp'
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
    plt.scatter(df.loc[df['n_id'] == outlet_node]['lon'], df.loc[df['n_id'] == outlet_node]['lat'], facecolors = 'None', edgecolors = 'r')
    for lat, lon, label in zip(clustergdf.geometry.y, clustergdf.geometry.x, clustergdf.Building):
        ax.annotate(label, xy=(lon, lat), xytext=(lon, lat))
    plt.show()
    

            
wb.save('v2_MINLP_STEP_Gravity.xls')



