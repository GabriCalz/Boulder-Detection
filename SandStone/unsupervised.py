#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script with some unsupervised methods to perform boulder identification
"""
import sys
sys.path.append('../')

#Imports that are needed
import os
import numpy as np
import open3d as o3d
import configSandStone as cfss
import pyransac3d as pyrsc 
import matplotlib.pyplot as plt
import generalFunctions as gf

class SandStoneClass(object):
    
    def __init__(self, pointcloudPoints):
        self.pointcloudPoints = pointcloudPoints
    
    def executePSR(self, threshold = cfss.Psr_threshold, depth = 9, eps=2, min_points=6):
        pointcloud = o3d.geometry.PointCloud()
        boulders_pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.pointcloudPoints)
        colours = np.tile([0,1,0], (len(self.pointcloudPoints),1))
        pointcloud.estimate_normals()
        
        gf.printDelimiter()
        print("Execution method is set to Poisson Surface Reconstruction")
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud, depth=depth)
        o3d.visualization.draw_geometries([mesh])

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)  
        
        
        emptyspots, boulders = [], []
        for index, point in enumerate(self.pointcloudPoints):
            query_point = o3d.core.Tensor([[point[0], point[1], point[2]]], dtype=o3d.core.Dtype.Float32)
            distance = scene.compute_distance(query_point)
            if distance > threshold:
                boulders.append(point)
                colours[index, :] = [1, 0, 0]
            else: emptyspots.append(point)
            
        boulders_pointcloud.points = o3d.utility.Vector3dVector(boulders)
        pointcloud.paint_uniform_color([1, 0, 0])
        labels = np.array(boulders_pointcloud.cluster_dbscan(eps, min_points))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        boulders_pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([boulders_pointcloud, pointcloud])
        
        myDataFrame = gf.getDataframe(np.matrix(boulders), np.matrix(emptyspots), cfss.columnNames_results)    
        myDataFrame.to_csv(os.path.join(cfss.results_folder, f"SS_section{cfss.mySurface.value + 1}.csv"))
        
        pointcloud = o3d.utility.Vector3dVector(colours)
        o3d.visualization.draw_geometries([pointcloud])
        print("Boulder Ratio: ", (len(boulders)/len(self.pointcloudPoints))*100)
    
    def executeRANSAC(self, threshold=cfss.Ransac_threshold, eps=1, min_points=6, allPoints = True, step = 25):
        print("Execution method is set to RANSAC")
        pointcloud = o3d.geometry.PointCloud()
        boulders_pointcloud = o3d.geometry.PointCloud()
        seabed_cloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.pointcloudPoints)
        pointcloud_boundings = gf.getBoundaries(self.pointcloudPoints)
       
        if allPoints == True:
            
            bestEquation, inliers = pointcloud.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
            seabed_cloud = pcd.select_by_index(inliers)
            seabed_cloud.paint_uniform_color([0, 1.0, 0])
            boulders_pointcloud = pcd.select_by_index(inliers, invert=True)
            boulders_pointcloud.paint_uniform_color([1.0, 0, 0])
            o3d.visualization.draw_geometries([seabed_cloud, boulders_pointcloud])
        
        else:
            
            subpointclouds_list = []
            for starty in np.arange(pointcloud_boundings[2], pointcloud_boundings[3], step):
                newpointcloud, pointcloudtoadd = [], o3d.geometry.PointCloud()
                for point in self.pointcloudPoints:
                    if point[1] >= starty and point[1] <= starty+step:
                        newpointcloud.append(point)
                pointcloudtoadd.points = o3d.utility.Vector3dVector(newpointcloud)
                if len(pointcloudtoadd.points) >= 3:
                    subpointclouds_list.append(pointcloudtoadd)
                            
            for pointc in subpointclouds_list:
                bestEquation, inliers = pointc.segment_plane(distance_threshold=threshold,
                                                  ransac_n=3,
                                                  num_iterations=1000)
                partseabed_cloud = pointc.select_by_index(inliers)
                partseabed_cloud.paint_uniform_color([0, 1.0, 0])
                partboulders_pointcloud = pointc.select_by_index(inliers, invert=True)
                partboulders_pointcloud.paint_uniform_color([1.0, 0, 0])
                boulders_pointcloud += partboulders_pointcloud
                seabed_cloud += partseabed_cloud
                o3d.visualization.draw_geometries([partseabed_cloud, partboulders_pointcloud])    
        
        labels = np.array(boulders_pointcloud.cluster_dbscan(eps, min_points))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        boulders_pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([boulders_pointcloud])

        myDataFrame = gf.getDataframe(boulders_pointcloud.points, seabed_cloud.points, cfss.columnNames_results)    
        myDataFrame.to_csv(os.path.join(cfss.results_folder, f"SS_section{cfss.mySurface.value + 1}.csv"))
        
        o3d.visualization.draw_geometries([seabed_cloud, boulders_pointcloud])
        print("Boulder Ratio: ", (len(boulders_pointcloud.points)/len(self.pointcloudPoints)*100))

if __name__ == "__main__":

    print("Loading the point cloud file")
    pointcloud = np.loadtxt(cfss.BalticSeaPC_file, skiprows = cfss.startIndexSection[cfss.mySurface.value], max_rows=cfss.Range_dimension)
    
    gf.printInfoPointCloud(pointcloud, f"Section {cfss.mySurface.value + 1}")
    gf.printDelimiter()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.visualization.draw_geometries([pcd])
    
    sandstone = SandStoneClass(pointcloud)
    if cfss.myMethod == cfss.UnsupervisedMethods.RANSAC:
        sandstone.executeRANSAC()
    else:
        sandstone.executePSR()
        
    gf.printDelimiter()            
    print("End of Script")