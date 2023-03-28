#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:03:59 2022

@author: weiqi
"""

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

#Imports that are needed
import numpy as np
import open3d as o3d
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from torch.utils.tensorboard import SummaryWriter

import configNorthShore as cf
import nsFunctions as fu
import generalFunctions as gf

writer = SummaryWriter()

class Engine_SupervisedClassifier(object):
    
    def __init__(self, aiMethod = cf.Ai_usedmethod, sectionCode_training = cf.section_training, sectionCode_testing = cf.section_testing):
        
        self.sectionCode_training = sectionCode_training
        self.sectionCode_testing = sectionCode_testing
        self.machineLearning_method = aiMethod
        
    def changeAIMethod(self, newMethod):
        self.machineLearning_method = newMethod
    
    def getSpecificEngine(self, index, k):
        if index == cf.AIAlgorithmEnum.RandomForestClassifier.value: return RandomForestRegressor(n_estimators = k, random_state = 42), "Creating model using the Random Forest Classifier" #30
        elif index == cf.AIAlgorithmEnum.KNNClassifier.value:        return KNeighborsClassifier(), "Creating model using the K-Neighbors Classifier"
        elif index == cf.AIAlgorithmEnum.MLPClassifier.value:        return MLPClassifier(hidden_layer_sizes= (11), max_iter=10000), "Creating model using the Multi-Layer Perceptron Classifier" 
        elif index == cf.AIAlgorithmEnum.SVCLinear.value:            return SVC(kernel="linear", C=1.8), "Creating model using the SVC Linear Classifier"
        elif index == cf.AIAlgorithmEnum.SVCGamma.value:             return SVC(gamma=2, C=5.9), "Creating model using the SVC Gamma Classifier"
        elif index == cf.AIAlgorithmEnum.GaussianProcessClass.value: return GaussianProcessClassifier(1.0 * RBF(1.0)), "Creating model using the Gaussian Process Classifier"
        elif index == cf.AIAlgorithmEnum.DecisionTreeClass.value:    return DecisionTreeClassifier(max_depth=5), "Creating model using the Decision Tree Classifier"
        elif index == cf.AIAlgorithmEnum.AdaBoostClass.value:        return AdaBoostClassifier(), "Creating model using the Adaboost Classifier"
        elif index == cf.AIAlgorithmEnum.GaussianNB.value:           return GaussianNB(), "Creating model using the Gaussian NB Classifier"
        elif index == cf.AIAlgorithmEnum.QuadraticDiscrAnaly.value:  return QuadraticDiscriminantAnalysis(), "Creating model using the Quadratic Discriminant Analysis Classifier"
                
    def trainingFunction(self, rf, index, k, training_split, sectionCode = cf.section_training):
        print("Section used for training: Section ", (sectionCode.value + 1))
        train_features, test_features, train_labels, test_labels = training_split
        predictions = rf.predict(test_features)
        
        preciseAccuracy, approximatedAccuracy, correctlyfoundAccuracy, f1score =  self.getSentiment(predictions, test_labels, "Training " + str(rf) + " on section " + str(sectionCode.value+1), k)
        
        writer.add_scalar("Precise classification accuracy", preciseAccuracy, k)
        writer.add_scalar("approximated accuracy", approximatedAccuracy, k)
        writer.add_scalar("correctly found accuracy", correctlyfoundAccuracy, k)
        writer.add_scalar("f1 accuracy", f1score, k)
        
        # if index == cf.AIAlgorithmEnum.RandomForestClassifier.value:
            
        #     print("Analysis of the importance of each Feature:")
        #     importances = list(rf.feature_importances_)
        #     feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(cf.features, importances)]
        #     feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        #     [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
            
        #     rf_most_important = RandomForestRegressor(n_estimators= 30, random_state=42)
        #     important_indices = [ cf.features.index(feature_importances[0][0]), cf.features.index(feature_importances[1][0])]
            
        #     train_important = train_features[:, important_indices]
        #     test_important = test_features[:, important_indices]
        #     rf_most_important.fit(train_important, train_labels)
        #     predictions = rf_most_important.predict(test_important)
            
            #self.getSentiment(predictions, test_labels, "Most important features for Forest Classifier")
            
            # second_figure = matplotlib.figure.Figure()
            # second_figure.plt.style.use('fivethirtyeight')
            # x_values = list(range(len(importances)))
            # second_figure.plt.bar(x_values, importances, orientation = 'vertical')
            # fl = cf.features
            # second_figure.plt.xticks(x_values, fl, rotation='vertical')
            # second_figure.plt.ylabel('Importance'); plt.xlabel('Variable'); 
            # second_figure.plt.get_current_fig_manager().canvas.manager.set_window_title("Importances")
            # second_figure.plt.show()
    
        
    def testingFunction(self, rf, sectionCode = cf.section_testing):
        
        section = sectionCode.value + 1
        
        print(f"Section used for testing: Section {section}")
        
        filepath = os.path.join(cf.SeabedSections_folder, f"Section{section}.csv")
        test_locations, test_features, test_labels, myTable = fu.tableReader(filepath)
        
        test_features = MinMaxScaler().fit_transform(test_features)  
        test_predictions = rf.predict(test_features)
        
        self.getSentiment(test_predictions, test_labels, f"Testing {rf} on section {section}")
        
        myTable["Label"] = np.around(test_predictions)
        pathresultFile = os.path.join(cf.results_folder, f"NS_section{section}.csv")
        myTable.to_csv(pathresultFile)
        print(f"Overwritten file: {pathresultFile}")
        gf.printDelimiter()
        
    def testAutoPick(self, rf):
        
        for file in sorted(os.listdir(cf.autopick_folder)):
            print(file)
            if "AutopickData" in file:
                
                print("Autopick file to be tested: ", file)
                
                filepath = os.path.join(cf.autopick_folder, file)
                test_locations, test_features, test_labels, myTable = fu.tableReader(filepath)
                
                test_features = MinMaxScaler().fit_transform(test_features)  
                test_predictions = rf.predict(test_features)
                
                self.getSentiment(test_predictions, test_labels, "Testing " + str(rf) + " on file " + file)
                
                gf.printDelimiter()
        
    def getSentiment(self, predictions, labels, titlePlot, k=0):
        
        preciseAccuracy, approximatedAccuracy = fu.classificationAccuracy(predictions, labels)
        truePositives, trueNegatives, falsePositives, falseNegatives, correctlyfoundAccuracy = fu.findConfusionMatrix(predictions, labels)
        f1score = fu.getF1Score(predictions, labels)
        
        print("Precise classification accuracy:", round(preciseAccuracy, 2), "%")
        print("Approximate classification accuracy:", approximatedAccuracy)
        print("ConfusionMatrix Accuracy", correctlyfoundAccuracy)
        print("f1 score", f1score)
        
        predictions = np.around(predictions)
        # disp = ConfusionMatrixDisplay( confusion_matrix = confusion_matrix(labels, predictions),
        #                                display_labels = cf.classification_classes)
        # disp.plot()
        # plt.get_current_fig_manager().canvas.manager.set_window_title(titlePlot)
        # plt.show()
        
        return round(preciseAccuracy, 2), approximatedAccuracy, correctlyfoundAccuracy, f1score
    
    def createAIEngine(self):
        
        filepath = os.path.join(cf.SeabedSections_folder, "Section" + str(self.sectionCode_training.value+1) + ".csv")
        locations_training, features_training, labels_training, myTable = fu.tableReader(filepath)

        #fu.computeShapiro(myTable)
            
        features_training = MinMaxScaler().fit_transform(features_training)
        training_split = train_test_split(features_training, labels_training, test_size=0.25, random_state=42)
        train_features, test_features, train_labels, test_labels = training_split
        
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', len(train_labels))
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', len(test_labels))
        
        gf.printDelimiter()
        
        algorithmSelected = np.zeros(len(cf.AIAlgorithmEnum)-1)
        if self.machineLearning_method == cf.AIAlgorithmEnum.Compare:
            algorithmSelected[:] = np.ones(len(cf.AIAlgorithmEnum)-1)
        else:
            algorithmSelected[self.machineLearning_method.value] = 1
            
        for index, value in enumerate(algorithmSelected):
            if value == 1:
                for k in range(10,11):
                    rf, description = self.getSpecificEngine(index,k)
                    print(description)
                    gf.printDelimiter()
                    rf.fit(train_features, train_labels);
                    self.trainingFunction(rf, index, k, training_split)
                    if cf.section_testing == cf.SurfaceCodeEnum.allZones:
                        for zone in [zone for zone in cf.SurfaceCodeEnum if zone != cf.section_training and zone != cf.SurfaceCodeEnum.allZones]:
                            self.testingFunction(rf, zone)
                    self.testAutoPick(rf)
                writer.flush()
                
if __name__ == "__main__":
    

    print("Supervised Machine Learning Algorithms")
    gf.printDelimiter()
    
    need = "analyseperf"
    if need == "CreateFeaturesFile":
        
        print("Creating Feature File")
        fu.createFeatureFile(cf.SurfaceCodeEnum.Zone4)
        fu.createFeatureFile(cf.SurfaceCodeEnum.Zone5)
        fu.createFeatureFile(cf.SurfaceCodeEnum.Zone6)
    else:
        
        print("Loading pointclouds for training and verification")
        Pointcloud_training = np.loadtxt(cf.BalticSeaPC_file, skiprows = cf.startIndexSection[cf.section_training.value], max_rows=cf.Range_dimension) 
        Pointcloud_testing  = np.loadtxt(cf.BalticSeaPC_file, skiprows = cf.startIndexSection[cf.section_testing.value], max_rows=cf.Range_dimension) 
        print(f"Training pointcloud [starting at {cf.startIndexSection[cf.section_training.value]} and long {cf.Range_dimension}] loaded.")
        print(f"Testing pointcloud [starting at {cf.startIndexSection[cf.section_testing.value]} and long {cf.Range_dimension}] loaded.")
    
        pc_training = o3d.geometry.PointCloud()
        pc_training.points = o3d.utility.Vector3dVector(Pointcloud_training)
        o3d.visualization.draw_geometries([pc_training])
        
        pc_testing = o3d.geometry.PointCloud()
        pc_testing.points = o3d.utility.Vector3dVector(Pointcloud_testing)
        o3d.visualization.draw_geometries([pc_testing])
    
        SeabedTraining_dimensions = gf.getBoundaries(Pointcloud_training)
        SeabedTesting_dimensions =  gf.getBoundaries(Pointcloud_testing)
    
        print("Seabed Training: ", SeabedTraining_dimensions[0], ":", SeabedTraining_dimensions[1], "&", 
                                   SeabedTraining_dimensions[2], ":", SeabedTraining_dimensions[3], "&", 
                                   SeabedTraining_dimensions[4], ":", SeabedTraining_dimensions[5])
        
        print("Seabed Testing: ", SeabedTesting_dimensions[0], ":", SeabedTesting_dimensions[1], "&", 
                                  SeabedTesting_dimensions[2], ":", SeabedTesting_dimensions[3], "&", 
                                  SeabedTesting_dimensions[4], ":", SeabedTesting_dimensions[5])
        
        gf.printDelimiter()
        print("Selected Method is ", cf.Ai_usedmethod)
        obj = Engine_SupervisedClassifier()
        obj.createAIEngine()
        print("End")