import numpy as np
import os
#from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import pvlib
import pandas as pd
from scipy import interpolate
import sys

def Getroominfor(filename):
    '''
    This function get information from the html file and sort each zone by layer.
    zoneinfor:[Zone_name,Zaxis,Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,ExteriorGrossArea,ExteriorWindowArea]
    Args:
        filename: html file
    Returns:
        layerAll: nxm zoneinfor list. n:zones number in this layer, m:layers number.
        roonum: room number integer
        cordall: n zoneinfor list. n:total zones number.
    '''
    # Read all lines of the html file
    htmllines = open(filename).readlines()

    # Initialize count and printflag variables
    count = 0
    printflag = False

    # Initialize lists for storing zone information
    cordall = []
    cord = []

    # Iterate through each line in the html file
    for line in htmllines:
        count += 1

        # Turn off the printflag after the 'Zone info' chart
        if 'Zone Internal Gains Nominal' in str(line):
            printflag = False

        # Extract information when the printflag is True
        if printflag:
            #Zone_name
            if (count - 35) % 32 == 0 and count != 3:
                linestr = str(line)
                cord.append(linestr[22:-6])
            #Zaxis
            if (count - 42) % 32 == 0 and count != 10:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Xmin
            if (count - 46) % 32 == 0 and count != 14:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Xmax
            if (count - 47) % 32 == 0 and count != 15:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Ymin
            if (count - 48) % 32 == 0 and count != 16:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Ymax
            if (count - 49) % 32 == 0 and count != 17:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Zmin
            if (count - 50) % 32 == 0 and count != 18:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Zmax
            if (count - 51) % 32 == 0 and count != 19:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #FloorArea
            if (count - 56) % 32 == 0 and count != 24:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #ExteriorNetArea
            if (count - 58) % 32 == 0 and count != 26:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #ExteriorWindowArea
            if (count - 59) % 32 == 0 and count != 27:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))

                # Append the current cord to cordall and reset cord
                cordall.append(cord)
                cord = []

        # Set printflag to True when 'Zone Information' is encountered in the line
        if 'Zone Information' in str(line):
            printflag = True
            count = 0

    # Calculate the total number of zones
    roomnum = len(cordall)

    # Sort cordall by Zaxis
    cordall.sort(key=lambda x: x[1])

    # Initialize layerAll, roomlist, and samelayer
    samelayer = cordall[0][1]
    roomlist = [cordall[0]]
    layerAll = []

    # Append an index to each zone in cordall
    cordall[0].append(0)

    # Sort each zone by layer
    for i in range(1, len(cordall)):
        cordall[i].append(i)

        # If the Zaxis value changes, add the current roomlist to layerAll
        # and reset the roomlist and samelayer variables
        if cordall[i][1] != samelayer:
            layerAll.append(roomlist)
            roomlist = [cordall[i]]
            samelayer = cordall[i][1]

            # If this is the last zone, append the roomlist to layerAll
            if i == len(cordall) - 1:
                layerAll.append(roomlist)
        else:
            # If the Zaxis value remains the same, add the zone to the roomlist
            roomlist.append(cordall[i])

            # If this is the last zone, append the roomlist to layerAll
            if i == len(cordall) - 1:
                layerAll.append(roomlist)
    return layerAll, roomnum, cordall



def checkconnect(room1min,room1max,room2min,room2max):
  '''
  This function check whether zones in the same layer are connected.
  '''
  if (room1min[0]>=room2min[0] and room1min[0]<=room2max[0] \
    and room1min[1]>=room2min[1] and room1min[1]<=room2max[1])\
    or(room1max[0]>=room2min[0] and room1max[0]<=room2max[0] \
    and room1max[1]>=room2min[1] and room1max[1]<=room2max[1]):
    return True
  return False


def checkconnect_layer(room1min,room1max,room2min,room2max):
  '''
  This function check whether zones in different layers are connected.
  '''
  if (room1min[0]>=room2min[0] and room1min[0]<room2max[0] \
    and room1min[1]>=room2min[1] and room1min[1]<room2max[1])\
    or(room1max[0]>room2min[0] and room1max[0]<=room2max[0] \
    and room1max[1]>room2min[1] and room1max[1]<=room2max[1]):
    return True
  return False

def Nfind_neighbor(roomnum,Layerall,U_Wall,SpecificHeat_avg):
  '''
  This function is for the building model.
  Args:
      roomnum: room number,
      Layerall: sorted layer list,
      U_Wall: U-factor for all walls,
      SpecificHeat_avg: Specific heat.
  Returns:
      dicRoom: map dictionary for neighbour,n+1 by n,
      Rtable: R table(n:roomnumber),
      Ctable: n by 1 C table(n:roomnumber),
      Windowtable: n by 1 Window table(n:roomnumber)
  '''
  # Initialize Rtable, Ctable, and Windowtable
  Rtable=np.zeros((roomnum,roomnum+1))
  Ctable=np.zeros(roomnum)
  Windowtable = np.zeros(roomnum)

  # Unpack U_Wall values
  Walltype=U_Wall[0]
  Floor =U_Wall[1]
  OutWall = U_Wall[2]
  OutRoof = U_Wall[3]
  Ceiling = U_Wall[4]
  Window = U_Wall[6]

  # Set air density value
  Air = 1.225 #kg/m^3

  # Initialize the dictionary for room neighbors
  dicRoom={}
  outind=roomnum

  # Iterate through each layer in Layerall
  for k in range(len(Layerall)):
    Layer_num=len(Layerall)
    cordall = Layerall[k]
    FloorRoom_num = len(cordall)

    # Check for neighbors in the layer above
    if k+1<Layer_num:
      nextcord=Layerall[k+1]
      for i in range(len(cordall)):
        for j in range(len(nextcord)):

          # Get zone coordinates for the current layer and the layer above
          x1min=[float(cordall[i][2]),float(cordall[i][4])]
          x1max=[float(cordall[i][3]),float(cordall[i][5])]
          x2min=[float(nextcord[j][2]),float(nextcord[j][4])]
          x2max=[float(nextcord[j][3]),float(nextcord[j][5])]

          # Check if zones are connected between layers
          if checkconnect_layer(x2min,x2max,x1min,x1max)or checkconnect_layer(x1min,x1max,x2min,x2max):

            # Calculate cross-sectional area
            crossarea=(min(x1max[1],x2max[1])-max(x1min[1],x2min[1]))*(min(x1max[0],x2max[0])-max(x1min[0],x2min[0]))

            # Calculate heat transfer coefficient (U) for connected zones
            U = crossarea*(Floor*Ceiling/(Floor+Ceiling))

            # Update Rtable for connected zones
            Rtable[nextcord[j][11],cordall[i][11]]=U
            Rtable[cordall[i][11],nextcord[j][11]]=U

            # Update the dictionary with connected zones
            if cordall[i][0] in dicRoom:
              dicRoom[cordall[i][0]].append(nextcord[j][11])
            else:
              dicRoom[cordall[i][0]]=[nextcord[j][11]]
            if nextcord[j][0] in dicRoom:
              dicRoom[nextcord[j][0]].append(cordall[i][11])
            else:
              dicRoom[nextcord[j][0]]=[cordall[i][11]]

    # Calculate heat capacity (C) and window area for each zone in the current layer
    for i in range(len(cordall)) :
      height = cordall[i][7]-cordall[i][6]
      xleng= (cordall[i][3]-cordall[i][2])
      yleng= cordall[i][5]-cordall[i][4]

      # Compute heat capacity (C) for the current zone
      C_room=SpecificHeat_avg*height*xleng*yleng*Air
      Ctable[cordall[i][11]]= C_room

      # Update Windowtable for the current zone
      Windowtable[cordall[i][11]]=cordall[i][10]

      # Update Rtable for exterior zones
      if cordall[i][9]>0 or (i==len(cordall)-1):
        if i==len(cordall)-1:
          Rtable[cordall[i][11],-1]= cordall[i][9]*OutWall+xleng*yleng*OutRoof+cordall[i][10]*Window


        else:
          Rtable[cordall[i][11],-1]= cordall[i][9]*OutWall+cordall[i][10]*Window

        # Update the dictionary
        if cordall[i][0] in dicRoom:
          dicRoom[cordall[i][0]].append(outind)
        else:
          dicRoom[cordall[i][0]]=[outind]

      # Check for neighbors within the same layer
      for j in range(i+1,FloorRoom_num):
        x1min=[float(cordall[i][2]),float(cordall[i][4])]
        x1max=[float(cordall[i][3]),float(cordall[i][5])]
        x2min=[float(cordall[j][2]),float(cordall[j][4])]
        x2max=[float(cordall[j][3]),float(cordall[j][5])]

        # Check if zones are connected within the same layer
        if checkconnect(x2min,x2max,x1min,x1max)or checkconnect(x1min,x1max,x2min,x2max):
          # Calculate the length of the shared wall
          length=np.sqrt((min(x1max[1],x2max[1])-max(x1min[1],x2min[1]))**2+(min(x1max[0],x2max[0])-max(x1min[0],x2min[0]))**2)

          # Calculate heat transfer coefficient (U) for connected zones
          U = height*length*Walltype

          # Update Rtable for connected zones
          Rtable[cordall[j][11],cordall[i][11]]=U
          Rtable[cordall[i][11],cordall[j][11]]=U

          # Update the dictionary with connected zones
          if cordall[i][0] in dicRoom:
            dicRoom[cordall[i][0]].append(cordall[j][11])
          else:
            dicRoom[cordall[i][0]]=[cordall[j][11]]
          if cordall[j][0] in dicRoom:
            dicRoom[cordall[j][0]].append(cordall[i][11])
          else:
            dicRoom[cordall[j][0]]=[cordall[i][11]]
  return dicRoom,Rtable,Ctable,Windowtable





def ParameterGenerator(Building,Weather,Location,U_Wall=0,Ground_Tp=[0,0,0,0,0,0,0,0,0,0,0,0],
                       shgc=0.252,shgc_weight=0.01,ground_weight=0.5,full_occ=0,
                       max_power=8000,AC_map=1,time_reso=3600,reward_gamma=[0.001,0.999],
                       target=20,activity_sch=np.ones(100000000)*1*120,temp_range=[-40,40],spacetype='continuous'):
    """
    This function could generate parameters from the selected building and temperature file for the env.

    Args:
      Building (str): Either name of a building in the predefined `Building_dic` or the file path to a htm file for building idf.
      Weather (str): Either name of a weather condition in the predefined `weather_dic` or the file path to an epw file.
      Location (str): Name of the location that matches an entry in `GroundTemp_dic`.
      U_Wall (list of floats): U-values (thermal transmittance) for different surfaces in the building in the order of [intwall, floor, outwall, roof, ceiling, groundfloor, window].
      Ground_Tp (float): Ground temperature when location is not in `GroundTemp_dic`.
      shgc (float): Solar Heat Gain Coefficient for the window. Default is 0.252.
      shgc_weight (float): Weight factor for extra loss of solar irradiance (ghi). Default is 0.01.
      ground_weight (float): Weight factor for extra loss of heat from ground. Default is 0.5.
      full_occ (numpy array): Shape (roomnum,1). Max number of people that can occupy each room. Default is all zeros.
      max_power (int): Maximum power output of a single HVAC unit, in watts. Default is 8000.
      AC_map (numpy array): Shape (roomnum,). Boolean map indicating presence (1) or absence (0) of AC in each room. Default is all ones.
      time_reso (int): Length of 1 timestep in seconds. Default is 3600 (1 hour).
      reward_gamma (list of two floats): [Energy penalty, temperature error penalty]. Default is [0.001,0.999].
      target (float or numpy array): Shape (roomnum,). Target temperature setpoints for each zone. Default is 22 degrees Celsius.
      activity_sch (numpy array): Shape (length of the simulation,). Activity schedule of people in the building in watts/person. Default is all 120.
      temp_range (list of two ints or floats): [Min temperature, max temperature], defining comfort range. Default is [-40,40] degrees Celsius.
      spacetype (str): Defines if it is a continuous or discrete space. Default is 'continuous'.

    Returns:
      Parameter (dict): Contains all parameters needed for environment initialization.
    """
    # Define dictionaries for Building, Ground Temperature, and Weather
    Building_dic={'ApartmentHighRise':['ASHRAE901_ApartmentHighRise_STD2019_Tucson.table.htm',[6.299,3.285,0.384,0.228,3.839,0.287,2.786]]
    ,'ApartmentMidRise':['ASHRAE901_ApartmentMidRise_STD2019_Tucson.table.htm',[6.299,3.285,0.384,0.228,3.839,0.287,2.786]]
    ,'Hospital':['ASHRAE901_Hospital_STD2019_Tucson.table.htm',[6.299,3.839,0.984,0.228,3.839,3.285,2.615]]
    ,'HotelLarge':['ASHRAE901_HotelLarge_STD2019_Tucson.table.htm',[6.299,0.228,0.984,0.228,0.228,2.705,2.615]]
    ,'HotelSmall':['ASHRAE901_HotelSmall_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,3.839,0.1573,2.615]]
    ,'OfficeLarge':['ASHRAE901_OfficeLarge_STD2019_Tucson.table.htm',[6.299,3.839,0.984,0.228,4.488,3.839,2.615]]
    ,'OfficeMedium':['ASHRAE901_OfficeMedium_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,4.488,0.319,2.615]]
    ,'OfficeSmall':['ASHRAE901_OfficeSmall_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,4.488,0.319,2.615]]
    ,'OutPatientHealthCare':['ASHRAE901_OutPatientHealthCare_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,3.839,0.5650E-02,2.615]]
    ,'RestaurantFastFood':['ASHRAE901_RestaurantFastFood_STD2019_Tucson.table.htm',[6.299,0.158,0.547,4.706,0.158,0.350,2.557]]
    ,'RestaurantSitDown':['ASHRAE901_RestaurantSitDown_STD2019_Tucson.table.htm',[6.299,0.158,0.514,4.706,0.158,0.194,2.557]]
    ,'RetailStandalone':['ASHRAE901_RetailStandalone_STD2019_Tucson.table.htm',[6.299,0.047,0.984,0.228,0.228,0.047,3.695]]
    ,'RetailStripmall':['ASHRAE901_RetailStripmall_STD2019_Tucson.table.htm',[6.299,0.1125,0.514,0.228,0.228,0.1125,3.695]]
    ,'SchoolPrimary':['ASHRAE901_SchoolPrimary_STD2019_Tucson.table.htm',[6.299,0.144,0.514,0.228,0.228,0.144,2.672]]
    ,'SchoolSecondar':['ASHRAE901_SchoolSecondary_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,3.839,0.144,2.672]]
    ,'Warehouse':['ASHRAE901_Warehouse_STD2019_Tucson.table.htm',[0.774,0.1926,1.044,0.5892,10.06,0.1926,2.557]]}
    GroundTemp_dic={'Albuquerque':[13.7,7.0,2.1,2.6,4.3,8.8,13.9,17.8,23.2,25.6,24.1,20.5],
            'Atlanta':[16.0,11.9,7.7,4.0,7.9,13.8,17.2,20.8,24.8,26.1,26.5,22.5],
            'Buffalo':[9.7,6.0,-2.2,-3.4,-4.2,2.7,7.5,13.7,18.6,22.0,20.7,16.5],
            'Denver':[7.1,3.0,-1.0,0.8,-0.2,4.8,6.1,13.7,22.2,22.7,21.7,18.5],
            'Dubai':[29.5,25.5,21.1,19.2,20.8,23.1,26.5,31.4,33.0,35.1,35.3,32.5],
            'ElPaso':[18.3,11.2,6.8,8.1,10.3,12.5,19.2,23.8,27.9,27.5,26.3,23.4],
            'Fairbanks':[-3.1,-17.7,-19.3,-17.6,-15.4,-10.3,0.7,10.6,16.0,16.9,14.2,6.7],
            'GreatFalls':[8.6,2.8,-4.1,-8.8,-2.2,0.3,6.7,10.1,16.5,20.6,19.2,14.7],
            'HoChiMinh':[26.9,26.7,26.0,26.4,27.5,28.3,29.2,29.0,28.9,27.2,27.5,27.6],
            'Honolulu':[26.2,24.8,23.7,22.5,22.8,23.2,23.8,25.2,25.9,26.9,27.1,26.9],
            'InternationalFalls':[5.4,-2.0,-14.6,-16.9,-11.5,-6.2,4.0,13.4,18.0,19.7,17.9,12.3],
            'NewDelhi':[25.1,19.6,14.5,13.4,17.0,22.4,29.1,33.0,33.6,31.7,30.0,28.7],
            'NewYork':[14.0,7.3,3.3,1.2,-0.2,5.6,10.9,16.1,21.7,25.0,24.8,19.9],
            'PortAngeles':[9.3,6.7,4.1,4.2,4.2,5.9,9.0,10.0,13.3,15.0,15.7,13.4],
            'Rochester':[7.4,-0.0,-7.6,-12.6,-7.7,0.3,7.0,14.2,19.2,20.9,20.0,15.4],
            'SanDiego':[18.8,14.3,13.6,13.2,13.3,12.6,15.3,15.6,17.7,19.4,19.7,18.5],
            'Seattle':[11.4,8.1,5.4,4.5,5.8,8.3,10.9,13.0,15.6,17.7,18.8,15.1],
            'Tampa':[24.2,18.9,15.7,13.6,15.5,17.1,21.2,26.9,27.6,27.9,27.4,26.2],
            'Tucson':[20.9,15.4,11.9,14.8,12.7,15.4,23.3,26.3,31.2,30.4,29.8,27.8]}
    weather_dic={'Very_Hot_Humid':'USA_HI_Honolulu.Intl.AP.911820_TMY3.epw',
            'Hot_Humid':'USA_FL_Tampa-MacDill.AFB.747880_TMY3.epw',
            'Hot_Dry':'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
            'Warm_Humid':'USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw',
            'Warm_Dry':'USA_TX_El.Paso.Intl.AP.722700_TMY3.epw',
            'Warm_Marine':'USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw',
            'Mixed_Humid':'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw',
            'Mixed_Dry':'USA_NM_Albuquerque.Intl.Sunport.723650_TMY3.epw',
            'Mixed_Marine':'USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw',
            'Cool_Humid':'USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw',
            'Cool_Dry':'USA_CO_Denver-Aurora-Buckley.AFB.724695_TMY3.epw',
            'Cool_Marine':'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
            'Cold_Humid':'USA_MN_Rochester.Intl.AP.726440_TMY3.epw',
            'Cold_Dry':'USA_MT_Great.Falls.Intl.AP.727750_TMY3.epw',
            'Very_Cold':'USA_MN_International.Falls.Intl.AP.727470_TMY3.epw',
            'Subarctic/Arctic':'USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw'}
    Carbon_dic={'ElPaso':[0.423,0.438,0.445,0.443,0.441,0.439,0.453,0.467,0.482,0.463,0.447,0.434,0.426,0.449,0.459,0.468,0.481,0.498,0.498,0.497,0.430,0.391,0.396,0.409]}
    
    Price_dic={'ElPaso':[0.035,0.033,0.032,0.029,0.031,0.034,0.037,0.039,0.041,0.036,0.033,0.032,0.030,0.029,0.029,0.027,0.028,0.031,0.030,0.032,0.032,0.033,0.035,0.035]}

    groundtemp=np.tile(GroundTemp_dic[Location], 31*24*2*3600//time_reso)
    carbonintensity=np.tile(Carbon_dic[Location], 31*24*3600//time_reso)
    electricityprice=np.tile(Price_dic[Location], 31*24*3600//time_reso)

    # Check if Building is in Building_dic, otherwise use Building as filename
    if Building not in Building_dic:
      filename = Building
    else:
      filename = os.path.join('envs','building','data', Building_dic[Building][0])
      U_Wall = Building_dic[Building][1]

    # Check if Weather is in weather_dic, otherwise use Weather as weatherfile
    if Weather not in weather_dic:
      weatherfile = [Weather,groundtemp]
    else:
      weatherfile = [os.path.join('envs','building','data', weather_dic[Weather]),groundtemp]

    # Get room information from the building file
    Layerall, roomnum, buildall  = Getroominfor(filename)
    #print("###############All Zones from Ground############")
    #for build in buildall:
    #  print (build[0],' [Zone index]: ',build[-1])
    #print("###################################################")


    # Read the weather data and interpolate temperature values
    data=pvlib.iotools.read_epw(weatherfile[0])
    oneyear=data[0]['temp_air']
    num_datapoint=len(oneyear)
    x = np.arange(0, num_datapoint)
    y = np.array(oneyear)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint-1, 1/3600*time_reso)
    outtempdatanew = f(xnew)

    # Read the weather data and interpolate GHI values to the new time resolution
    oneyearrad=data[0]['ghi']
    x = np.arange(0, num_datapoint)
    y = np.array(oneyearrad)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint-1, 1/3600*time_reso)
    solardatanew = f(xnew)

    # Define constants and calculate SHGC
    Air=1.225 #kg/m^3
    SpecificHeat_avg = 1000
    SHGC=shgc*shgc_weight*(max(data[0]['ghi'])/(abs(data[1]['TZ'])/60))

    # Find neighboring rooms, resistance and capacitance tables, and window properties
    dicRoom,Rtable,Ctable,Windowtable=Nfind_neighbor(roomnum,Layerall,U_Wall,SpecificHeat_avg)

    # Initialize connectivity matrix and ground connection list
    connectmap=np.zeros((roomnum,roomnum+1))
    RCtable=Rtable/np.array([Ctable]).T
    ground_connectlist=np.zeros((roomnum,1)) #list to see which room connects to the ground
    groundrooms=Layerall[0] #the first layer connects to the ground

    # Assign ground connection values and populate the connectivity matrix
    for room in groundrooms:
      ground_connectlist[room[11]]=room[8]*U_Wall[5]*ground_weight #for those rooms, assign 1/R table by floor area and u factor

    for i in range(len(buildall)):
        connect_list = dicRoom[buildall[i][0]]

        for number in connect_list:
            connectmap[i][number] = 1

    # Calculate occupancy, AC weight, weighted connection map, and non-linear term
    people_full= np.zeros((roomnum,1))+np.array([full_occ]).T
    ACweight=np.diag(np.zeros(roomnum)+AC_map)* max_power
    weightcmap = np.concatenate((people_full,ground_connectlist,np.zeros((roomnum, 1)), ACweight, np.array([Windowtable*SHGC]).T), axis=-1)/np.array([Ctable]).T
    nonlinear= people_full/np.array([Ctable]).T

    # Store parameters in a dictionary for the simulation
    Parameter = {}
    Parameter['OutTemp'] = outtempdatanew
    Parameter['connectmap'] = connectmap
    Parameter['RCtable'] = RCtable
    Parameter['roomnum'] = roomnum
    Parameter['weightcmap'] = weightcmap
    Parameter['target'] = np.zeros(roomnum) + target
    Parameter['gamma'] = reward_gamma
    Parameter['time_resolution']=time_reso
    Parameter['ghi'] = solardatanew/(abs(data[1]['TZ'])/60)/ (max(data[0]['ghi'])/(abs(data[1]['TZ'])/60))
    Parameter['GroundTemp'] = weatherfile[1]
    Parameter['Occupancy'] = activity_sch
    Parameter['ACmap']= np.zeros(roomnum)+AC_map
    Parameter['max_power'] = max_power
    Parameter['nonlinear'] = nonlinear
    Parameter['temp_range'] = temp_range
    Parameter['spacetype'] = spacetype
    Parameter['CarbonIntensity'] = carbonintensity
    Parameter['ElectricityPrice'] = electricityprice

    return Parameter



