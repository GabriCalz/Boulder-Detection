#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:12:28 2022

@author: weiqi
"""

import os
from enum import Enum

marineIdentification_folder = os.path.dirname(os.getcwd())
sandStone_folder = os.path.dirname(os.path.realpath(__file__))

BalticSeaHumanAnn_file = os.path.join(marineIdentification_folder, "HOW04_Nearshore_MBES_Contacts.xlsx")
BalticSeaPC_file = os.path.join(marineIdentification_folder, "HOW04_Nearshore_DTM_025M.txt")
SeabedSections_folder = os.path.join(marineIdentification_folder, "SharedData/processedData")
results_folder = os.path.join(sandStone_folder, "SandStoneResults")
autopick_folder = os.path.join(marineIdentification_folder, "SharedData/processedData/AutopickData")

columnNames_results = ["Easting Coord", "Northing Coord", "Depth", "Label"]

classification_classes = ["Empty", "Boulder"]

class SurfaceCodeEnum(Enum):
    Zone1 = 0
    Zone2 = 1
    Zone3 = 2
    Zone4 = 3
    Zone5 = 4
    Zone6 = 5
    allZones = 6
    
class UnsupervisedMethods(Enum):
    RANSAC = 1
    PoissonSurfaceReconstruction = 2
    
startIndexSection = [0, 1000000, 2000000, 11000000, 12000000, 13000000, 5000000]
Range_dimension = 500000  
Ransac_threshold = 0.15 #0.256
Psr_threshold = 0.15

mySurface = SurfaceCodeEnum.Zone1
myMethod = UnsupervisedMethods.PoissonSurfaceReconstruction

