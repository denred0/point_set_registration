import open3d as o3d
import numpy as np
import copy


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud],
                                      width=1080,
                                      height=720)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      width=1080,
                                      height=720)


def icp(pcd_source, pcd_target, threshold_list, trans_init, type, max_iteration):
    if type == 'point':
        print("Apply point-to-point ICP")
    elif type == 'plane':
        print("Apply point-to-plane ICP")

    fitness_list = []
    for threshold in threshold_list:
        threshold = np.round(threshold, 2)
        if type == 'point':
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_source, pcd_target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        elif type == 'plane':
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_source, pcd_target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

        print(f"Threshold: {threshold}, "
              f"fitness: {reg_p2p.fitness}, "
              f"inlier_rmse: {reg_p2p.inlier_rmse}, "
              f"corr_set: {np.asarray(reg_p2p.correspondence_set).shape[0]}")
        fitness_list.append(reg_p2p.fitness)

    best_threshold = threshold_list[fitness_list.index(max(fitness_list))]
    print(f"Best threshold: {best_threshold}")

    if type == 'point':
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target, best_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    elif type == 'plane':
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target, best_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.transformation, np.asarray(reg_p2p.correspondence_set)


def remove_outliers(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Points removed: {len(np.asarray(pcd.points)) - len(np.asarray(pcd.select_by_index(ind).points))}")
    return pcd.select_by_index(ind)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, voxel_size, transformation):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def prepare_scan_ar(scan_number):
    source_cloud = f'data/scans/Scan{scan_number}.txt'
    with open(source_cloud) as f:
        lines = f.readlines()

    coords = lines[1].split(",")
    x = np.asarray(coords[0::3])
    y = np.asarray(coords[1::3])
    z = np.asarray(coords[2::3])

    min_count = np.min([len(x), len(y), len(z)])
    x = x[:min_count]
    y = y[:min_count]
    z = z[:min_count]

    points = np.vstack((x, y, z)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = remove_outliers(pcd, nb_neighbors=10, std_ratio=0.2)

    return pcd


if __name__ == "__main__":
    # target cloud
    target_cloud = 'data/target_scene/target.pcd'
    pcd_target = o3d.io.read_point_cloud(target_cloud)
    pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    o3d.visualization.draw_geometries([pcd_target], width=1080, height=720)

    # prepare all AR scans, remove outliers
    scan_clouds = []
    for i in range(1, 11):
        pcd = prepare_scan_ar(i)
        scan_clouds.append(pcd)

    # Point-set registration all AR scans
    trans_init = np.asarray([[0.862, 0.011, -0.507, 2],
                             [-0.139, 0.967, -0.215, 0.3],
                             [0.487, 0.255, 0.835, 2], [0.0, 0.0, 0.0, 1.0]])
    result_scan_cloud = scan_clouds[0]
    for i in range(1, 10):
        psource = scan_clouds[i]
        ptarget = result_scan_cloud

        threshold_list = np.linspace(0.01, 0.02, 5)
        # point-to-point algorithm
        point_transformation, correspondence_set = icp(psource,
                                                       ptarget,
                                                       threshold_list,
                                                       trans_init,
                                                       type='point',
                                                       max_iteration=2000)
        psource.transform(point_transformation)
        result_scan_cloud += psource.select_by_index(correspondence_set[:, 0])

    # invert Z-coordinate
    np.asarray(result_scan_cloud.points)[:, 2] = np.asarray(result_scan_cloud.points)[:, 2] * (-1)
    pcd_scan = o3d.geometry.PointCloud()
    pcd_scan.points = o3d.utility.Vector3dVector(np.asarray(result_scan_cloud.points))
    pcd_scan.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    o3d.visualization.draw_geometries([pcd_scan], width=1080, height=720)

    # remove alone points
    pcd_scan = remove_outliers(pcd_scan, nb_neighbors=20, std_ratio=1.5)
    o3d.visualization.draw_geometries([pcd_scan], width=1080, height=720)

    # show initial registration
    draw_registration_result(pcd_scan, pcd_target, trans_init)

    # start point-to-point algorithm with different thresholds
    threshold_list = np.linspace(0.01, 0.5, 10)
    point_transformation, correspondence_set = icp(pcd_scan,
                                                   pcd_target,
                                                   threshold_list,
                                                   trans_init,
                                                   type='point',
                                                   max_iteration=2000)
    draw_registration_result(pcd_scan, pcd_target, point_transformation)

    # start point-to-plane algorithm with different thresholds
    plane_transformation, correspondence_set = icp(pcd_scan,
                                                   pcd_target,
                                                   threshold_list,
                                                   trans_init,
                                                   type='plane',
                                                   max_iteration=2000)
    draw_registration_result(pcd_scan, pcd_target, plane_transformation)

    #
    voxel_size = 0.05
    source_down, source_fpfh = preprocess_point_cloud(pcd_scan, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_target, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_icp = refine_registration(pcd_scan, pcd_target, voxel_size, result_ransac.transformation)
    print(result_icp)
    draw_registration_result(pcd_scan, pcd_target, result_icp.transformation)
