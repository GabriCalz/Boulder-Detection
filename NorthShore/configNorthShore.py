#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:03:43 2022

@author: weiqi
"""

from enum import Enum
import os

marineIdentification_folder = os.path.dirname(os.getcwd())
northshore_folder = os.path.dirname(os.path.realpath(__file__))

BalticSeaHumanAnn_file = os.path.join(marineIdentification_folder, "HOW04_Nearshore_MBES_Contacts.xlsx")
BalticSeaPC_file = os.path.join(marineIdentification_folder, "HOW04_Nearshore_DTM_025M.txt")

SeabedSections_folder = os.path.join(marineIdentification_folder, "processedData")
results_folder = os.path.join(northshore_folder, "NorthShoreResults")
autopick_folder = os.path.join(marineIdentification_folder, "processedData/AutopickData")

dataframe_header = ["Easting Coord", "Northing Coord", "Depth", "Standard Deviation 0.5", "Standard Deviation 1",	
                    "Standard Deviation 2", "Standard Deviation 3", "Mean 0.5", "Mean 1", "Mean 2", "Mean 3", 
                    "Dz 0.5", "Dz 1", "Dz 2", "dz 3", "Dp 1", "Dp 2", "Dp 3", "Linearity", "Planarity", "Sphericity",
                    "Omnivariance", "Anisotropy", "Change of Curvature", "Eigenentropy", "Scattering", "Label"]
locations = ["Easting Coord", "Northing Coord", "Depth"]
features = ["Depth", "Standard Deviation 0.5", "Standard Deviation 1", "Standard Deviation 2", "Standard Deviation 3", 
            "Mean 0.5", "Mean 1", "Mean 2", "Mean 3", "Dz 0.5", "Dz 1", "Dz 2", "dz 3", "Dp 1", "Dp 2", "Dp 3", 
            "Linearity", "Planarity", "Sphericity","Omnivariance", "Anisotropy", "Change of Curvature", "Eigenentropy", "Scattering"]
classification_classes = ["Empty", "Boulder"]

class AIAlgorithmEnum(Enum):
    RandomForestClassifier = 0
    KNNClassifier = 1
    MLPClassifier = 2 #good
    SVCLinear = 3 #good
    SVCGamma = 4 #Very good but depends on the zone
    GaussianProcessClass = 5 #Good but there could be better (still better than GaussianNb)
    DecisionTreeClass = 6 #very good
    AdaBoostClass = 7 #Very good
    GaussianNB = 8 #Good but there could be better, but it works (max error 14%)
    QuadraticDiscrAnaly = 9 #Good in Training, but a total mess in the test phase
    Compare = 10
    
class SurfaceCodeEnum(Enum):
    Zone1 = 0
    Zone2 = 1
    Zone3 = 2
    Zone4 = 3
    Zone5 = 4
    Zone6 = 5
    allZones = 6
    
startIndexSection = [0, 1000000, 2000000, 11000000, 12000000, 13000000, 5000000]
Range_dimension = 500000  
Ransac_threshold = 0.256

Ai_usedmethod = AIAlgorithmEnum.RandomForestClassifier
section_training = SurfaceCodeEnum.Zone1
section_testing = SurfaceCodeEnum.allZones

    
