#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:04:35 2022

@author: weiqi
"""
import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import configNorthShore as cf
import open3d as o3d
import random as rnd
from scipy.stats import shapiro
import statistics as st
from sklearn.decomposition import PCA
import generalFunctions as gf

def colorPickedAreasInPointCloud(sectionCode):
    
    myTable = pd.read_csv(os.path.join(cf.SeabedSections_folder, "Section" + str(sectionCode.value) + ".csv"))
    locationsAndLabel_training = myTable[[cf.locations + "Label"]]  
    pointcloud = np.loadtxt(cf.BalticSeaPC_file, skiprows=cf.startIndexSection[sectionCode.value], max_rows=cf.Range_dimension) 
    
    # featuresWithoutBoulders = pd.read_csv(self.getNameOfProperFile(FileTypeEnum.FreeSpotsLocationsFile, TrainingZone)).to_numpy()
    # featuresWithBoulders = pd.read_csv(self.getNameOfProperFile(FileTypeEnum.BoulderLocationsFile, TrainingZone)).to_numpy()       

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pointCloudTree = o3d.geometry.KDTreeFlann(pcd)
   
    
    for element in locationsAndLabel_training:
        [k, idx, _] = pointCloudTree.search_radius_vector_3d(element[1:3], 1)
        if element[-1] == 1: np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        else: np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]


    o3d.visualization.draw_geometries([pcd])
    
def computeFeatures(center, status, typeOfSet, point_cloud, numb):
    
    print("Computing Features on the Point Cloud with Code: ", status, "-", str(typeOfSet))
    
    pcdFeatures = o3d.geometry.PointCloud()
    pcdFeatures.points = o3d.utility.Vector3dVector(point_cloud)
    pointCloudTree = o3d.geometry.KDTreeFlann(pcdFeatures)
    
    # path = os.path.join(cf.SeabedSections_folder, "Zone"+str(numb)+"/section" + str(status) + ".pcd")
    # open(path, 'w').close()
    # o3d.io.write_point_cloud(path, pcdFeatures)
    
    [k05, idx05, _] = pointCloudTree.search_radius_vector_3d(center, 0.5)
    [k1, idx1, _] = pointCloudTree.search_radius_vector_3d(center, 1)
    [k2, idx2, _] = pointCloudTree.search_radius_vector_3d(center, 2)
    [k3, idx3, _] = pointCloudTree.search_radius_vector_3d(center, 3)
    
    pointCloud05 = np.asarray(pcdFeatures.points)[idx05[1:], :] 
    pointCloud1 =  np.asarray(pcdFeatures.points)[idx1[1:], :] 
    pointCloud2 =  np.asarray(pcdFeatures.points)[idx2[1:], :] 
    pointCloud3 =  np.asarray(pcdFeatures.points)[idx3[1:], :] 
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pointCloud3)
    
    # if typeOfSet == "boulder": 
    #     path = os.path.join(cf.SeabedSections_folder, "Zone"+str(numb)+"/boulder" + str(status) + ".pcd")
    #     open(path, 'w').close()
    #     o3d.io.write_point_cloud(path, pcd2)
    # else: 
    #     path = os.path.join(cf.SeabedSections_folder, "Zone"+str(numb)+"/empty" + str(status) + ".pcd")
    #     open(path, 'w').close()
    #     o3d.io.write_point_cloud(path, pcd2)
    STD = [st.stdev(pointCloud05[:,2]), st.stdev(pointCloud1[:,2]), st.stdev(pointCloud2[:,2]), st.stdev(pointCloud3[:,2]) ]
    Mean = [st.mean(pointCloud05[:,2]), st.mean(pointCloud1[:,2]), st.mean(pointCloud2[:,2]), st.mean(pointCloud3[:,2]) ]
    dz = [ center[2] - min(pointCloud05[:,2]),  center[2] - min(pointCloud1[:,2]),  center[2] - min(pointCloud2[:,2]),  center[2] - min(pointCloud3[:,2])]
    
    if len(pointCloud05) <=2:
        omnivariance = 0
        linearity = 0
        planarity = 0
        sphericity = 0
        anisotropy = 0
        surface_variation = 0
    else:
        
        pcaPointCloud05 = PCA().fit(pointCloud05)
        eigenvalues_3D = pcaPointCloud05.singular_values_ ** 2
        norm_eigenvalues_3D = eigenvalues_3D / np.sum(eigenvalues_3D) 
        omnivariance = np.prod(norm_eigenvalues_3D) ** (1 / 3)
        linearity = (norm_eigenvalues_3D[0] - norm_eigenvalues_3D[1]) / norm_eigenvalues_3D[0]
        planarity = (norm_eigenvalues_3D[1] - norm_eigenvalues_3D[2]) / norm_eigenvalues_3D[0]
        sphericity = norm_eigenvalues_3D[2]/norm_eigenvalues_3D[0]
        anisotropy = (norm_eigenvalues_3D[0] - norm_eigenvalues_3D[2]) / norm_eigenvalues_3D[0]
        surface_variation = norm_eigenvalues_3D[2]
        scattering = norm_eigenvalues_3D[2] / norm_eigenvalues_3D[0]
        nonnull_eig = norm_eigenvalues_3D[norm_eigenvalues_3D > 0]
        eigenentropy = -1 * np.sum(nonnull_eig * np.log(nonnull_eig))
    
    pcdFeatures.points = o3d.utility.Vector3dVector(pointCloud1)
    bestEquation1, _ = pcdFeatures.segment_plane(distance_threshold=cf.Ransac_threshold,
                                     ransac_n=3,
                                     num_iterations=1000)
    pcdFeatures.points = o3d.utility.Vector3dVector(pointCloud2)
    bestEquation2, _ = pcdFeatures.segment_plane(distance_threshold=cf.Ransac_threshold,
                                     ransac_n=3,
                                     num_iterations=1000)
    pcdFeatures.points = o3d.utility.Vector3dVector(pointCloud3)
    bestEquation3, _ = pcdFeatures.segment_plane(distance_threshold=cf.Ransac_threshold,
                                     ransac_n=3,
                                     num_iterations=1000)

    distance1 = gf.distancePointFromPlane(center[0], center[1], center[2], bestEquation1[0], 
                                     bestEquation1[1], bestEquation1[2], bestEquation1[3])
    distance2 = gf.distancePointFromPlane(center[0], center[1], center[2], bestEquation2[0], 
                                      bestEquation2[1], bestEquation2[2], bestEquation2[3])
    distance3 = gf.distancePointFromPlane(center[0], center[1], center[2], bestEquation3[0], 
                                      bestEquation3[1], bestEquation3[2], bestEquation3[3])
    
    if typeOfSet == "boulder": label = 1
    else: label = 0
    
    featuresPacked = (center[0], center[1], center[2], STD[0], STD[1], STD[2], STD[3], Mean[0], Mean[1], Mean[2], Mean[3], 
                      dz[0], dz[1], dz[2], dz[3], distance1, distance2, distance3, linearity, 
                      planarity, sphericity, omnivariance, anisotropy, surface_variation, eigenentropy,
                      scattering, label)
    
    return featuresPacked

def createFeatureFile(sectionCode):
    
    pointcloud = np.loadtxt(cf.BalticSeaPC_file, skiprows = cf.startIndexSection[sectionCode.value], max_rows=cf.Range_dimension)
    boundaries = gf.getBoundaries(pointcloud)
    oceaninfinityAnnotated = np.array(pd.read_excel(cf.BalticSeaHumanAnn_file))
    manualPick = oceaninfinityAnnotated[1757:6629, 1:4]
    myFeatureTable = np.empty((0, len(cf.dataframe_header)))
    counter = 0

    for element in manualPick:
        if gf.checkPointInsideBoundaries(element, boundaries):
            counter += 1
            feat = computeFeatures(element.tolist(), counter, "boulder", pointcloud, sectionCode.value+1)
            myFeatureTable = np.append(myFeatureTable, np.asarray(feat).reshape((1, len(feat))) , axis=0)
            
    counterOfBoulders = counter
    counterOfFree = 0      
    while counter >= 1:
        point = pointcloud[rnd.randrange(len(pointcloud)), :]
        tooclose = False
        for boulder in manualPick:
            if gf.insideSphere(boulder, point, 2):
                tooclose = True
        if (not tooclose) and gf.checkPointInsideBoundaries(point, boundaries):
            counterOfFree += 1 
            feat = computeFeatures(point.tolist(), counter, "Empty", pointcloud, sectionCode.value+1)
            myFeatureTable = np.append(myFeatureTable, np.asarray(feat).reshape((1, len(feat))) , axis=0)
            counter -= 1
            
    print(counterOfBoulders, ":", counterOfFree)
    featureDF = pd.DataFrame(myFeatureTable)
    featureDF.columns = cf.dataframe_header
    featureDF.to_csv(os.path.join(cf.SeabedSections_folder, "Section" + str(sectionCode.value+1) + ".csv"))
    
def saveSectionOfPointCloud(readFrom, whereToSave, sectionCode):
    
    pointcloud = np.loadtxt(readFrom, skiprows=cf.startIndexSection[sectionCode.value], max_rows=cf.Range_dimension)   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud(whereToSave, pcd)
    o3d.visualization.draw_geometries([pcd])
    
def classificationAccuracy(predictionVector, labelsVector):
    try:
        preciseAccuracy = 1 - np.mean(abs(predictionVector - labelsVector))
        predictionApproximated = np.around(predictionVector)
        approximatedAccuracy = np.sum(predictionApproximated == labelsVector) / len(labelsVector)
    except: print("error in division")
    return preciseAccuracy*100, approximatedAccuracy

def tableReader(fileName):
    
    myTable = pd.read_csv(fileName)
    locations = myTable[cf.locations].to_numpy() 
    features = myTable[cf.features].to_numpy() 
    labels = [int(element) for element in myTable["Label"].to_list()]
    return locations, features, labels, myTable

def findConfusionMatrix(predictions, labels):
    predictions = np.around(predictions)
    labels = np.around(labels)
    try:
        truePositives, trueNegatives, falsePositives, falseNegatives = 0,0,0,0
        for i in range( len(predictions) ):
            if predictions[i] == 1 and labels[i] == 1: truePositives += 1
            elif predictions[i] == 0 and labels[i] == 0: trueNegatives += 1
            elif predictions[i] == 1 and labels[i] == 0: falsePositives += 1
            elif predictions[i] == 0 and labels[i] == 1: falseNegatives += 1
            
        correctlyfoundAccuracy = (truePositives + trueNegatives)/len(predictions)
        print(f"TP: {truePositives}")
        print(f"TN: {trueNegatives}")
        print(f"FP: {falsePositives}")
        print(f"FN: {falseNegatives}")
    except: print("error in division")
    return truePositives, trueNegatives, falsePositives, falseNegatives, correctlyfoundAccuracy

def getAreaUnderCurve(predictions, labels):
    try:
        truePositives, trueNegatives, falsePositives, falseNegatives, correctlyfoundAccuracy = findConfusionMatrix(predictions, labels)
        truePositiveRate = truePositives/(falseNegatives + truePositives)
        trueNegativeRate = trueNegatives/(trueNegatives + falsePositives)
    except: print("error in division")
    return truePositiveRate, trueNegativeRate

def getF1Score(predictions, labels):
    f1score = 0
    try:
        truePositives, trueNegatives, falsePositives, falseNegatives, correctlyfoundAccuracy = findConfusionMatrix(predictions, labels)
        precision = truePositives/(truePositives+falsePositives)
        recall = truePositives/(truePositives+falseNegatives)
        f1score = 2*1/((1/precision)+(1/recall))
    except: print("error in division")
    return f1score

def computeShapiro(myTable):
    for feature in cf.features:
        status, probability = shapiro(myTable[feature])
        print(f"Feature {feature} has status: {status} and probability: {probability}")
        if probability>0.05: print("----->Probably gaussian")
    
def createAutoPickDataFile(start, end):
    pointcloud = np.loadtxt(cf.BalticSeaPC_file)
    oceaninfinityAnnotated = np.array(pd.read_excel(cf.BalticSeaHumanAnn_file))
    autoPick = oceaninfinityAnnotated[start:end, 1:4]
    myFeatureTable = np.empty((0, len(cf.dataframe_header)))
    counter = 0

    for element in autoPick:
        counter += 1
        feat = computeFeatures(element.tolist(), counter, "boulder", pointcloud, 2)
        myFeatureTable = np.append(myFeatureTable, np.asarray(feat).reshape((1, len(feat))) , axis=0)

    featureDF = pd.DataFrame(myFeatureTable)
    featureDF.columns = cf.dataframe_header
    featureDF.to_csv(os.path.join(cf.SeabedSections_folder, "AutopickData6.csv"))



    