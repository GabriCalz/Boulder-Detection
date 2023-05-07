#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:12:28 2022

@author: weiqi
"""

import os
from enum import Enum

General_folder = os.path.join(os.path.dirname(os.getcwd()), 'General')
SandStone_folder = os.path.dirname(os.path.realpath(__file__))
Results_folder = os.path.join(SandStone_folder, "SandStoneResults")
folders = [General_folder, SandStone_folder, Results_folder]

BalticSeaAnnotated_file = os.path.join(General_folder, "HOW04_Nearshore_MBES_Contacts.xlsx")
BalticSeaPointCloud_file = os.path.join(General_folder, "HOW04_Nearshore_DTM_025M.txt")
files = [BalticSeaPointCloud_file, BalticSeaAnnotated_file]

# SeabedSections_folder = os.path.join(General_folder, "SharedData/processedData")
# autopick_folder = os.path.join(General_folder, "SharedData/processedData/AutopickData")

print("Checking on files and folders...")
foundAllData = True
for folder in folders:
    if not os.path.isdir(folder):
        print(f"Status Folder {folder} : Not Found")
        foundAllData = False
for file in files:
    if not os.path.isfile(file):
        print(f"Status File {file} : Not Found")
        foundAllData = False
if foundAllData:
    print("All files and folders are available.")

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
    BallPivoting = 3
    AlphaShapes = 4


startIndexSection = [0, 1000000, 2000000, 11000000, 12000000, 13000000, 5000000]
Range_dimension = 500000

mySurface = SurfaceCodeEnum.Zone1
myMethod = UnsupervisedMethods.AlphaShapes
