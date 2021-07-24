import numpy as np
from numpy.core.numeric import identity
import open3d as o3d
import sys
import time
import math
from utility.file import join, dirname, get_file_list, get_rgbd_file_lists, read_poses_from_log, write_poses_to_log
from utility.visualization import draw_registration_result
from .make_fragments import read_rgbd_image
from .register_fragments import preprocess_point_cloud, multiscale_icp, register_point_cloud_fpfh
from .optimize_posegraph import optimize_posegraph_for_scene_multiview, optimize_posegraph_for_refined_scene_multiview
from .refine_registration import draw_registration_result_original_color, local_refinement, matching_result
from .make_fragments import run as make_fragments_run

class matching_result:

    def __init__(self, s, t, trans):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans
        self.infomation = np.identity(6)
        self.fitness = 0


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, config):
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "fgr":
        result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 3,
            [
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                1000000, 0.999))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)), 0)
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)), 0)
    return (True, result.transformation, information, result.fitness)


def register_point_cloud_pair(ply_file_names_multiview, ply_file_names, s, t, config):
    n_files = len(ply_file_names)
    if s < n_files:
        print("reading %s ..." % ply_file_names[s])
        source = o3d.io.read_point_cloud(ply_file_names[s])
    else:
        print("reading %s ..." % ply_file_names_multiview[s-n_files])
        source = o3d.io.read_point_cloud(ply_file_names_multiview[s-n_files])
    if t < n_files:
        print("reading %s ..." % ply_file_names[t])
        target = o3d.io.read_point_cloud(ply_file_names[t])
    else:
        print("reading %s ..." % ply_file_names_multiview[t-n_files])
        target = o3d.io.read_point_cloud(ply_file_names_multiview[t-n_files])
    (source_down, source_fpfh) = preprocess_point_cloud(source, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target, config)
    (success, transformation, information, fitness) = \
        register_point_cloud_fpfh(
            source_down, target_down, source_fpfh, target_fpfh, config)
    if not success:
        return (False, np.identity(4), np.identity(6), 0)
    if config["debug_mode"]:
        print(transformation)
        print(information)
    return (True, transformation, information, fitness)


def multiscale_icp(source,
                   target,
                   voxel_size,
                   max_iter,
                   config,
                   init_transformation=np.identity(4)):
    current_transformation = init_transformation
    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = config["voxel_size"] * 1.4
        print("voxel_size {}".format(voxel_size[scale]))
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])
        if config["icp_method"] == "point_to_point":
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iter))
        else:
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            if config["icp_method"] == "point_to_plane":
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=iter))
            if config["icp_method"] == "color":
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=iter))
        current_transformation = result_icp.transformation
        if i == len(max_iter) - 1:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, voxel_size[scale] * 1.4,
                result_icp.transformation)

    return (result_icp.transformation, information_matrix, result_icp.fitness)


def local_refinement(source, target, transformation_init, config):
    voxel_size = config["voxel_size"]
    (transformation, information, fitness) = \
        multiscale_icp(
        source, target,
        [voxel_size, voxel_size/2.0, voxel_size/4.0], [50, 30, 14],
        config, transformation_init)
    return (transformation, information, fitness)


def register_point_cloud_pair_refine(ply_file_names_multiview, ply_file_names, s, t, transformation_init,
                                     config):
    n_files = len(ply_file_names)
    if s < n_files:
        print("reading %s ..." % ply_file_names[s])
        source = o3d.io.read_point_cloud(ply_file_names[s])
    else:
        print("reading %s ..." % ply_file_names_multiview[s-n_files])
        source = o3d.io.read_point_cloud(ply_file_names_multiview[s-n_files])
    if t < n_files:
        print("reading %s ..." % ply_file_names[t])
        target = o3d.io.read_point_cloud(ply_file_names[t])
    else:
        print("reading %s ..." % ply_file_names_multiview[t-n_files])
        target = o3d.io.read_point_cloud(ply_file_names_multiview[t-n_files])

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target,
                                                transformation_init)

    (transformation, information, fitness) = \
        local_refinement(source, target, transformation_init, config)

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target, transformation)
        print(transformation)
        print(information)
    return (transformation, information, fitness)


def update_posegraph_for_scene(s, t, transformation, information, pose_graph, is_node=False):
    if is_node: 
        odometry_inv = np.dot(pose_graph.nodes[t].pose, np.linalg.inv(transformation))
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s,
                                                     t,
                                                     transformation,
                                                     information,
                                                     uncertain=True))
    return pose_graph


def make_posegraph_for_scene(ply_file_names_multiview, ply_file_names, config):
    odometry = np.identity(4)
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_refined_posegraph_optimized"]))
    pose_graph_multiview = o3d.pipelines.registration.PoseGraph(pose_graph)
    odometry = np.identity(4)

    n_files_multiview = len(ply_file_names_multiview)
    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files_multiview):
        for t in range(n_files):
            matching_results[t] = matching_result(s+n_files, t, None)

        if config["python_multi_threading"] == True:
            from joblib import Parallel, delayed
            import multiprocessing
            import subprocess
            MAX_THREAD = min(multiprocessing.cpu_count(),
                             max(len(matching_results), 1))
            results = Parallel(n_jobs=MAX_THREAD)(delayed(
                register_point_cloud_pair)(ply_file_names_multiview, ply_file_names, matching_results[r].s,
                                           matching_results[r].t, config)
                for r in matching_results)
            for i, r in enumerate(matching_results):
                matching_results[r].success = results[i][0]
                matching_results[r].transformation = results[i][1]
                matching_results[r].information = results[i][2]
                matching_results[r].fitness = results[i][3]
        else:
            for r in matching_results:
                (matching_results[r].success, matching_results[r].transformation,
                 matching_results[r].information, matching_results[r].fitness) = \
                    register_point_cloud_pair(ply_file_names_multiview, ply_file_names,
                                              matching_results[r].s, matching_results[r].t, config)
        i_best = 0
        best = 0
        for r in matching_results:
            if matching_results[r].fitness > best:
                i_best = r
                best = matching_results[r].fitness
        for r in matching_results:
            if matching_results[r].success:
                pose_graph = update_posegraph_for_scene(
                    matching_results[r].s, matching_results[r].t,
                    matching_results[r].transformation,
                    matching_results[r].information, pose_graph_multiview, i_best == r)
    print(pose_graph_multiview)
    o3d.io.write_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_multiview"]),
        pose_graph_multiview)


def make_posegraph_for_refined_scene(ply_file_names_multiview, ply_file_names, config):
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_multiview_optimized"]))
    n_files_multiview = len(ply_file_names_multiview)
    n_files = len(ply_file_names)
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id

        if s < n_files:
            continue

        transformation_init = edge.transformation
        matching_results[(s - n_files) * n_files + t] = \
            matching_result(s, t, transformation_init)

    if config["python_multi_threading"] == True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(pose_graph.edges), 1))
        results = Parallel(n_jobs=MAX_THREAD)(
            delayed(register_point_cloud_pair_refine)(
                ply_file_names_multiview, ply_file_names, matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation, config)
            for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].transformation = results[i][0]
            matching_results[r].information = results[i][1]
            matching_results[r].fitness = results[i][2]
    else:
        for r in matching_results:
            (matching_results[r].transformation,
             matching_results[r].information,
             matching_results[r].fitness) = \
                register_point_cloud_pair_refine(ply_file_names_multiview, ply_file_names,
                                                 matching_results[r].s, matching_results[r].t,
                                                 matching_results[r].transformation, config)

    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_refined_posegraph_optimized"]))
    pose_graph_new = o3d.pipelines.registration.PoseGraph(pose_graph)

    best = [(0, 0) for _ in range(n_files_multiview)]
    for r in matching_results:
        s = r // n_files
        if matching_results[r].fitness > best[s][1]:
            best[s] = r, matching_results[r].fitness
    
    for r, _ in best:
        pose_graph_new = update_posegraph_for_scene(
            matching_results[r].s, matching_results[r].t,
            matching_results[r].transformation, matching_results[r].information, pose_graph_new, True)
    o3d.io.write_pose_graph(
        join(config["path_dataset"],
             config["template_refined_posegraph_multiview"]),
        pose_graph_new)


def scalable_integrate_rgb_frames(path_dataset, intrinsic, config):
    poses = []
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    n_files = len(color_files)
    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))
    for view in range(4):
        [color_files1, depth_files1] = get_rgbd_file_lists(
            join(path_dataset, f"multiview{view}"))
        color_files += color_files1
        depth_files += depth_files1
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_global_posegraph_multiview_optimized"]))
    print(len(pose_graph_fragment.nodes))

    frame_id_abs = 0
    n_nodes = len(pose_graph_fragment.nodes)
    for fragment_id in range(n_nodes):
        if fragment_id < n_fragments:
            frame_id_abs = n_files
            continue
            pose_graph_rgbd = o3d.io.read_pose_graph(
                join(path_dataset,
                     config["template_fragment_posegraph_optimized"] % fragment_id))
        else:
            pose_graph_rgbd = o3d.io.read_pose_graph(
                join(path_dataset,
                     config["template_fragment_posegraph_multiview_optimized"] % (fragment_id-n_fragments, 0)))

        for frame_id in range(len(pose_graph_rgbd.nodes)):
            print(
                "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
                (fragment_id, n_nodes-1, frame_id_abs, frame_id+1,
                 len(pose_graph_rgbd.nodes)))
            rgbd = read_rgbd_image(color_files[frame_id_abs],
                                   depth_files[frame_id_abs], False, config)
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            poses.append(pose)
            frame_id_abs += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])

    mesh_name = join(path_dataset, config["template_global_mesh_multiview"])
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

    traj_name = join(path_dataset, config["template_global_traj_multiview"])
    write_poses_to_log(traj_name, poses)


def integrate_fragments(path_dataset, intrinsic, config):
    device = o3d.core.Device(config['device'])

    # Load RGBD
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)

    # Load extrinsics
    trajectory = read_poses_from_log(
        join(path_dataset, config["template_global_traj"]))

    n_files = len(color_files)

    n_fragments = int(
        math.ceil(float(n_files) / config['n_frames_per_fragment']))
    i = 0
    for fragment_id in range(n_fragments):
        # Setup volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config["tsdf_cubic_size"] / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        for frame_id in range(config['n_frames_per_fragment']):
            if i == n_files:
                break
            rgbd = read_rgbd_image(color_files[i],
                                    depth_files[i], False, config)

            start = time.time()
            volume.integrate(rgbd, intrinsic, np.linalg.inv(trajectory[i]))
            end = time.time()
            print('Integration {:04d}/{:04d} takes {:.3f} ms'.format(
                i, n_files, (end - start) * 1000.0))
            i += 1

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd_name = join(path_dataset,
                        config["template_fragment_pointcloud"].replace("fragments", "scene/fragments") % fragment_id)
        o3d.io.write_point_cloud(pcd_name, pcd, False, True)


def run(config):
    steps = [True, True, True, True]
    #steps = [False, False, False, True]
    path_dataset = config['path_dataset']

    # Load intrinsics
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # step 1
    if steps[0]:
        for view in range(4):
            config_view = config.copy()
            config_view["path_dataset"] = join(config_view["path_dataset"], f"multiview{view}")
            make_fragments_run(config_view)

    ply_file_names = get_file_list(
        join(path_dataset, config["folder_fragment"]), ".ply")
    ply_file_names_multiview = []
    for view in range(4):
        ply_file_names_multiview += get_file_list(
            join(path_dataset, f"multiview{view}", config["folder_fragment"]), ".ply")

    # step 2
    if steps[1]:
        print("register multiview fragments.")
        make_posegraph_for_scene(
            ply_file_names_multiview, ply_file_names, config)
        optimize_posegraph_for_scene_multiview(path_dataset, config)

    # step 3
    if steps[2]:
        print("refine rough registration of fragments.")
        make_posegraph_for_refined_scene(
            ply_file_names_multiview, ply_file_names, config)
        optimize_posegraph_for_refined_scene_multiview(path_dataset, config)

    # step 4
    if steps[3]:
        print("integrate the whole RGBD sequence using estimated camera pose.")
        scalable_integrate_rgb_frames(path_dataset, intrinsic, config)




