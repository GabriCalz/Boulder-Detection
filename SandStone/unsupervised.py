#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script with some unsupervised methods to perform boulder identification
"""
# import sys
# sys.path.append('../')

# Imports that are needed
# import os
import numpy as np
import open3d as o3d

import config as cf
import SandStone_definition as df
from General import generalFunctions as gf


gf.printDelimiter()
gf.printDelimiter()
print(f"Unsupervised Methods Script\nAlgorithm Name : {cf.myMethod.name}\nLoading the point cloud file")
startIndexOfSection = cf.startIndexSection[cf.mySurface.value]
lengthOfSection = cf.Range_dimension
PointCloud_points = np.loadtxt(cf.BalticSeaPointCloud_file, skiprows=startIndexOfSection, max_rows=lengthOfSection)
print(f"Point Cloud Section indexes -> {startIndexOfSection} : {startIndexOfSection + lengthOfSection}")
gf.printInfoPointCloud(PointCloud_points, f"Section {cf.mySurface.value + 1}")
gf.printDelimiter()

# Generation of the Point Cloud Object
PointCloud = o3d.geometry.PointCloud()
PointCloud.points = o3d.utility.Vector3dVector(PointCloud_points)
o3d.visualization.draw_geometries([PointCloud])

# Call of unsupervised methods
sandstone = df.SandStoneClass(PointCloud_points)
if cf.myMethod == cf.UnsupervisedMethods.RANSAC:
    sandstone.identify_with_ransac(downsample_voxel_size=0.5,
                                   ransac_threshold=0.10,
                                   clustering_eps=1,
                                   clustering_min_points=6,
                                   ransac_mode_all_points=True,
                                   sections_step=25)
elif cf.myMethod == cf.UnsupervisedMethods.PoissonSurfaceReconstruction:
    sandstone.identify_with_psr(downsample_voxel_size=0.5,
                                psr_threshold=0.15,
                                poisson_depth=9,
                                clustering_eps=2,
                                clustering_min_points= 6)
elif cf.myMethod == cf.UnsupervisedMethods.BallPivoting:
    sandstone.identify_with_ball_pivoting(downsample_voxel_size=1,
                                          ball_pivoting_threshold=0.15,
                                          radii_ball_pivoting=[0.5, 0.01, 0.02, 0.04],
                                          clustering_eps=2,
                                          clustering_min_points=6)
elif cf.myMethod == cf.UnsupervisedMethods.AlphaShapes:
    sandstone.identify_with_alpha_shapes(downsample_voxel_size=0.07,
                                         alpha_threshold=0.15,
                                         alpha_alpha_shapes=0.500,
                                         clustering_eps=2,
                                         clustering_min_points=6)
else:
    print("Error")


print("End of Script")
