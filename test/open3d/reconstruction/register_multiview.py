import numpy as np
from numpy.core.numeric import identity
import open3d as o3d
import sys
import math
from utility.file import join, dirname, get_file_list, get_rgbd_file_lists, write_poses_to_log
from utility.visualization import draw_registration_result
from .make_fragments import read_rgbd_image
from .register_fragments import preprocess_point_cloud, multiscale_icp, register_point_cloud_fpfh
from .optimize_posegraph import optimize_posegraph_for_scene_multiview, optimize_posegraph_for_refined_scene_multiview
from .refine_registration import draw_registration_result_original_color, local_refinement, matching_result


def compute_initial_registration(s, t, source_down, target_down, source_fpfh,
                                 target_fpfh, config):
    (success, transformation,
        information) = register_point_cloud_fpfh(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                config)
    if not success:
        print("No reasonable solution. Skip this pair")
        return (False, np.identity(4), np.zeros((6, 6)))
    print(transformation)

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)
    return (True, transformation, information)


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
    (success, transformation, information) = \
        compute_initial_registration(
        s, t, source_down, target_down,
        source_fpfh, target_fpfh, config)
    if not success:
        return (False, np.identity(4), np.identity(6))
    if config["debug_mode"]:
        print(transformation)
        print(information)
    return (True, transformation, information)


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

    (transformation, information) = \
        local_refinement(source, target, transformation_init, config)

    if config["debug_mode"]:
        draw_registration_result_original_color(source, target, transformation)
        print(transformation)
        print(information)
    return (transformation, information)


def update_posegraph_for_scene(s, t, transformation, information, odometry,
                               pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
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
    return (odometry, pose_graph)


def make_posegraph_for_scene(ply_file_names_multiview, ply_file_names, config):
    odometry = np.identity(4)
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_refined_posegraph_optimized"]))
    #pose_graph_multiview = o3d.pipelines.registration.PoseGraph(pose_graph)
    pose_graph_multiview = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph_multiview.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_files_multiview = len(ply_file_names_multiview)
    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files+n_files_multiview):
        for t in range(n_files+n_files_multiview):
            # if s < n_files and t < n_files:
            #     continue
            matching_results[s * (n_files+n_files_multiview) + t] = matching_result(s, t, odometry)

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
    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation,
             matching_results[r].information) = \
                register_point_cloud_pair(ply_file_names_multiview, ply_file_names,
                                          matching_results[r].s, matching_results[r].t, config)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegraph_for_scene(
                matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation,
                matching_results[r].information, odometry, pose_graph_multiview)
    o3d.io.write_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_multiview"]),
        pose_graph_multiview)


def make_posegraph_for_refined_scene(ply_file_names_multiview, ply_file_names, config):
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_multiview_optimized"]))

    n_files_multiview = len(ply_file_names_multiview)
    n_files = len(pose_graph.nodes)
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id

        # if s < n_files and t < n_files:
        #     continue

        transformation_init = edge.transformation
        matching_results[s * (n_files+n_files_multiview) + t] = \
            matching_result(s, t, transformation_init)

    if config["python_multi_threading"] == True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(pose_graph.edges), 1))
        results = Parallel(n_jobs=MAX_THREAD)(
            delayed(register_point_cloud_pair_refine)(
                ply_file_names_multiview, ply_file_names, matching_results[
                    r].s, matching_results[r].t,
                matching_results[r].transformation, config)
            for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].transformation = results[i][0]
            matching_results[r].information = results[i][1]
    else:
        for r in matching_results:
            (matching_results[r].transformation,
             matching_results[r].information) = \
                register_point_cloud_pair_refine(ply_file_names_multiview, ply_file_names,
                                                 matching_results[r].s, matching_results[r].t,
                                                 matching_results[r].transformation, config)

    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_refined_posegraph_optimized"]))
    #pose_graph_new = o3d.pipelines.registration.PoseGraph(pose_graph)
    pose_graph_new = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph_new.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    for r in matching_results:
        (odometry, pose_graph_new) = update_posegraph_for_scene(
            matching_results[r].s, matching_results[r].t,
            matching_results[r].transformation, matching_results[r].information,
            odometry, pose_graph_new)
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
        join(path_dataset, config["template_refined_posegraph_multiview_optimized"]))
    print(len(pose_graph_fragment.nodes))

    frame_id_abs = 0
    n_nodes = len(pose_graph_fragment.nodes)
    for fragment_id in range(n_nodes):
        if fragment_id < n_fragments:
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


def run(config):
    steps = [True, True, True, True]
    path_dataset = config['path_dataset']

    # step 1

    # step 2
    if steps[1]:
        print("register multiview fragments.")
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        ply_file_names = get_file_list(
            join(path_dataset, config["folder_fragment"]), ".ply")
        ply_file_names_multiview = []
        for view in range(4):
            ply_file_names_multiview += get_file_list(
                join(path_dataset, f"multiview{view}", config["folder_fragment"]), ".ply")
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
        if config["path_intrinsic"]:
            intrinsic = o3d.io.read_pinhole_camera_intrinsic(
                config["path_intrinsic"])
        else:
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        scalable_integrate_rgb_frames(path_dataset, intrinsic, config)
