#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:02:38 2022

@author: weiqi
"""
import math 
import numpy as np
import pandas as pd
import open3d as o3d


def distancePointFromPlane(x, y, z, A, B, C, D):
    return abs(A*x+B*y+C*z+D)/math.sqrt(A**2+B**2+C**2)

def insideSphere(point, center, radius):
    return math.sqrt( (point[0]-center[0])**2 +
                      (point[1]-center[1])**2 + 
                      (point[2]-center[2])**2   ) <= radius

def printInfoPointCloud(pointcloud, title):
    lx, ux, ly, uy, lz, uz = getBoundaries(pointcloud)
    numberofelements = len(pointcloud)
    print(f"{title}-> {lx}:{ux} & {ly}:{uy} & {lz}:{uz}")
    print(f"with total number of element: {numberofelements}")
    return lx, ux, ly, uy, lz, uz, numberofelements

def getBoundaries(pointcloud):
    return (np.min(pointcloud[:, 0]), np.max(pointcloud[:, 0]), np.min(pointcloud[:, 1]), np.max(pointcloud[:, 1]),
           np.min(pointcloud[:, 2]), np.max(pointcloud[:, 2]) )  

def checkPointInsideBoundaries(point, boundaries):
    return (boundaries[0] <= point[0] <= boundaries[1] and 
           boundaries[2] <= point[1] <= boundaries[3] and 
           boundaries[4] <= point[2] <= boundaries[5])

def printDelimiter():
    print("-------------------------------------------------------")

def getDataframe(boulders, emptyspots, column_names):
    
    boulderTable = np.append(np.matrix(boulders), np.ones((len(boulders),1)), axis=1)
    emptyspotsTable = np.append(np.matrix(emptyspots), np.zeros((len(emptyspots),1)), axis=1)
    data = np.append(boulderTable, emptyspotsTable, axis=0)
    return pd.DataFrame( data=data, columns = column_names)
    
def partitionPointCloud(filename, startIndexSection):
    j = 0
    for rangestart in startIndexSection:  
        for i in range(rangestart, rangestart+500000, 25000):
            print("Section that begins in ", i, " and terminates at ", i + 25000)
            Pointcloud_training = np.loadtxt(filename, skiprows = i, max_rows=25000) 
            pc_training = o3d.geometry.PointCloud()
            pc_training.points = o3d.utility.Vector3dVector(Pointcloud_training)
            filename = "/home/weiqi/Skrivbord/sections/area" + str(j) + ".pcd"
            o3d.io.write_point_cloud(filename, pc_training)
            j = j + 1
