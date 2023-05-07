# Imports that are needed
import os
import numpy as np
import open3d as o3d
# import pyransac3d as pyrsc
import matplotlib.pyplot as plt

from SandStone import config as cf
from General import generalFunctions as gf


class SandStoneClass(object):

    def __init__(self, points):
        self.PointCloud_points = points

    @staticmethod
    def downsample(PointCloud, voxel_size):
        print(f"Downsample the point cloud with a voxel of {voxel_size}")
        ReducedPointCloud = PointCloud.voxel_down_sample(voxel_size=voxel_size)
        print(
            f"Original Point Cloud -> number of points : {len(PointCloud.points)}\nDownsampled Point Cloud -> number of points : {len(ReducedPointCloud.points)}")
        return ReducedPointCloud

    def identify_with_alpha_shapes(self, downsample_voxel_size, alpha_threshold, alpha_alpha_shapes, clustering_eps,
                                   clustering_min_points):

        gf.printDelimiter()
        print("Execution method is set to Alpha Shapes")

        PointCloud = o3d.geometry.PointCloud()
        PointCloud.points = o3d.utility.Vector3dVector(self.PointCloud_points)
        o3d.visualization.draw_geometries([PointCloud])
        colours = np.tile([0, 1, 0], (len(self.PointCloud_points), 1))

        # Downsample the Point Cloud
        PointCloud = self.downsample(PointCloud, downsample_voxel_size)
        o3d.visualization.draw_geometries([PointCloud])

        # Apply Poisson Surface Reconstruction
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(PointCloud)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(PointCloud, alpha_alpha_shapes,
                                                                                        tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        # Create a scene and add the triangle mesh
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        emptyspots, boulders = [], []
        for index, point in enumerate(self.PointCloud_points):
            query_point = o3d.core.Tensor([[point[0], point[1], point[2]]], dtype=o3d.core.Dtype.Float32)
            distance = scene.compute_distance(query_point)  # Compute unsigned distance
            if distance > alpha_threshold:
                boulders.append(point)
                colours[index, :] = [1, 0, 0]
            else:
                emptyspots.append(point)

        PointCloud_boulders = o3d.geometry.PointCloud()
        PointCloud_boulders.points = o3d.utility.Vector3dVector(boulders)
        o3d.visualization.draw_geometries([PointCloud_boulders, PointCloud])
        PointCloud.paint_uniform_color([1, 0, 0])

        # Create clusters and show them
        self.create_clusters(PointCloud, PointCloud_boulders, clustering_eps, clustering_min_points)

        # Save results in a file
        self.save_data(np.matrix(boulders), np.matrix(emptyspots))
        print("Boulder Ratio: ", (len(boulders) / len(self.PointCloud_points)) * 100)

    def identify_with_ball_pivoting(self, downsample_voxel_size, ball_pivoting_threshold, radii_ball_pivoting,
                                    clustering_eps, clustering_min_points):

        gf.printDelimiter()
        print("Execution method is set to Ball Pivoting")

        PointCloud = o3d.geometry.PointCloud()
        PointCloud.points = o3d.utility.Vector3dVector(self.PointCloud_points)
        o3d.visualization.draw_geometries([PointCloud])
        colours = np.tile([0, 1, 0], (len(self.PointCloud_points), 1))

        # Fixing normals for each point in the Point Cloud Object
        PointCloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        PointCloud.estimate_normals()
        # PointCloud.orient_normals_consistent_tangent_plane(50)
        o3d.visualization.draw_geometries([PointCloud], point_show_normal=True)

        # Downsample the Point Cloud
        PointCloud = self.downsample(PointCloud, downsample_voxel_size)
        o3d.visualization.draw_geometries([PointCloud])

        # Apply Poisson Surface Reconstruction
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(PointCloud, o3d.utility.DoubleVector(
            radii_ball_pivoting))
        o3d.visualization.draw_geometries([mesh])

        # Create a scene and add the triangle mesh
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        emptyspots, boulders = [], []
        for index, point in enumerate(self.PointCloud_points):
            query_point = o3d.core.Tensor([[point[0], point[1], point[2]]], dtype=o3d.core.Dtype.Float32)
            distance = scene.compute_distance(query_point)  # Compute unsigned distance
            if distance > ball_pivoting_threshold:
                boulders.append(point)
                colours[index, :] = [1, 0, 0]
            else:
                emptyspots.append(point)

        PointCloud_boulders = o3d.geometry.PointCloud()
        PointCloud_boulders.points = o3d.utility.Vector3dVector(boulders)
        o3d.visualization.draw_geometries([PointCloud_boulders, PointCloud])
        PointCloud.paint_uniform_color([1, 0, 0])

        # Create clusters and show them
        self.create_clusters(PointCloud, PointCloud_boulders, clustering_eps, clustering_min_points)

        # Save results in a file
        self.save_data(np.matrix(boulders), np.matrix(emptyspots))
        print("Boulder Ratio: ", (len(boulders) / len(self.PointCloud_points)) * 100)

    def identify_with_psr(self, downsample_voxel_size, psr_threshold, poisson_depth, clustering_eps,
                          clustering_min_points):

        gf.printDelimiter()
        print("Execution method is set to Poisson Surface Reconstruction")

        PointCloud = o3d.geometry.PointCloud()
        PointCloud.points = o3d.utility.Vector3dVector(self.PointCloud_points)
        o3d.visualization.draw_geometries([PointCloud])
        colours = np.tile([0, 1, 0], (len(self.PointCloud_points), 1))

        # Fixing normals for each point in the Point Cloud Object
        PointCloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        PointCloud.estimate_normals()
        # PointCloud.orient_normals_consistent_tangent_plane(50)
        o3d.visualization.draw_geometries([PointCloud], point_show_normal=True)

        # Downsample the Point Cloud
        PointCloud = self.downsample(PointCloud, downsample_voxel_size)
        o3d.visualization.draw_geometries([PointCloud])

        # Apply Poisson Surface Reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=PointCloud, depth=poisson_depth)
        o3d.visualization.draw_geometries([mesh])

        # Print visualization of densities
        densities = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        o3d.visualization.draw_geometries([density_mesh])

        # Removal of low-density points
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(mesh)
        o3d.visualization.draw_geometries([mesh])

        # Create a scene and add the triangle mesh
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        emptyspots, boulders = [], []
        for index, point in enumerate(self.PointCloud_points):
            query_point = o3d.core.Tensor([[point[0], point[1], point[2]]], dtype=o3d.core.Dtype.Float32)
            distance = scene.compute_distance(query_point)  # Compute unsigned distance
            if distance > psr_threshold:
                boulders.append(point)
                colours[index, :] = [1, 0, 0]
            else:
                emptyspots.append(point)

        PointCloud_boulders = o3d.geometry.PointCloud()
        PointCloud_boulders.points = o3d.utility.Vector3dVector(boulders)
        o3d.visualization.draw_geometries([PointCloud_boulders, PointCloud])
        PointCloud.paint_uniform_color([1, 0, 0])

        # Create clusters and show them
        self.create_clusters(PointCloud, PointCloud_boulders, clustering_eps, clustering_min_points)

        # Save results in a file
        self.save_data(np.matrix(boulders), np.matrix(emptyspots))
        print("Boulder Ratio: ", (len(boulders) / len(self.PointCloud_points)) * 100)

    def identify_with_ransac(self, downsample_voxel_size, ransac_threshold, clustering_eps, clustering_min_points,
                             ransac_mode_all_points, sections_step):

        gf.printDelimiter()
        print("Execution method is set to RANSAC")

        PointCloud = o3d.geometry.PointCloud()
        PointCloud.points = o3d.utility.Vector3dVector(self.PointCloud_points)
        PointCloud_boundaries = gf.getBoundaries(self.PointCloud_points)

        # Downsample the Point Cloud
        PointCloud = self.downsample(PointCloud, downsample_voxel_size)
        o3d.visualization.draw_geometries([PointCloud])

        if ransac_mode_all_points:

            best_equation, inliers = PointCloud.segment_plane(distance_threshold=ransac_threshold,
                                                              ransac_n=3,
                                                              num_iterations=1000)
            PointCloud_seabed = PointCloud.select_by_index(inliers)
            PointCloud_seabed.paint_uniform_color([0, 1.0, 0])
            PointCloud_boulders = PointCloud.select_by_index(inliers, invert=True)
            PointCloud_boulders.paint_uniform_color([1.0, 0, 0])

            self.create_clusters(PointCloud_seabed, PointCloud_boulders, clustering_eps, clustering_min_points)
            self.save_data(PointCloud_boulders.points, PointCloud_seabed.points)

        else:

            subregions_list = []
            for start_y_coord in np.arange(PointCloud_boundaries[2], PointCloud_boundaries[3], sections_step):
                pointcloud_new = []
                for point in self.PointCloud_points:
                    if start_y_coord <= point[1] <= start_y_coord + sections_step:
                        pointcloud_new.append(point)

                if len(pointcloud_new) >= 3:
                    pointcloudtoadd = o3d.geometry.PointCloud()
                    pointcloudtoadd.points = o3d.utility.Vector3dVector(pointcloud_new)
                    subregions_list.append(pointcloudtoadd)

            for element in subregions_list:
                best_equation, inliers = element.segment_plane(distance_threshold=ransac_threshold,
                                                               ransac_n=3,
                                                               num_iterations=1000)
                PointCloud_seabed_part = element.select_by_index(inliers)
                PointCloud_seabed_part.paint_uniform_color([0, 1.0, 0])
                PointCloud_boulder_part = element.select_by_index(inliers, invert=True)
                PointCloud_boulder_part.paint_uniform_color([1.0, 0, 0])
                if len(PointCloud_boulder_part.points) > clustering_min_points:
                    self.create_clusters(PointCloud_seabed_part, PointCloud_boulder_part, clustering_eps,
                                         clustering_min_points)
                numberOfTotalPointInPart = len(PointCloud_boulder_part.points) + len(PointCloud_seabed_part.points)
                print(f"Boulder Ratio : {100 * len(PointCloud_boulder_part.points) / numberOfTotalPointInPart}")

    def create_clusters(self, PointCloud, PointCloud_boulders, eps, min_points):

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(PointCloud_boulders.cluster_dbscan(eps, min_points, print_progress=True))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        PointCloud_boulders.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([PointCloud_boulders, PointCloud])

    def save_data(self, points_boulders, points_seabed):

        myDataFrame = gf.getDataframe(points_boulders, points_seabed, cf.columnNames_results)
        myDataFrame.to_csv(os.path.join(cf.Results_folder, f"SS_section{cf.mySurface.value + 1}.csv"))

