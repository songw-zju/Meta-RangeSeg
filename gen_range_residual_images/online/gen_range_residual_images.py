import numpy as np
import torch
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils import load_poses, load_calib, load_files, load_vertex


def range_projection(current_vertex, proj_H=64, proj_W=2048, fov_up=3.0, fov_down=-25.0, max_range=50.0, min_range=2.0):
    """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns:
        proj_vertex: each pixel contains the corresponding point (x, y, z, depth)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > min_range) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > min_range) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x_orig = np.copy(proj_x)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y_orig = np.copy(proj_y)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_xyz = np.full((proj_H, proj_W, 3), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_mask = np.zeros((proj_H, proj_W),
                         dtype=np.int32)

    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z]).T
    proj_intensity[proj_y, proj_x] = intensity
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)

    return proj_range, proj_xyz, proj_intensity, proj_mask, proj_x_orig, proj_y_orig, current_vertex


def gen_range_residual_images(seq_idx, frame_idx, num_last_n):
    height = 64
    width = 2048
    fov_up = 3.0
    fov_down = -25.0
    max_range = 50.0
    min_range = 2.0
    # specify parameters
    num_frames = -1  # number of frames for training, -1 uses all frames
    debug = False  # plot images
    normalize = True  # normalize/scale the difference with corresponding range value
    num_last_n = num_last_n  # use the last n frame to calculate the difference image

    # load poses
    pose_file = './semantic_kitti/dataset/sequences/'+seq_idx+'/poses.txt'
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    calib_file = './semantic_kitti/dataset/sequences/'+seq_idx+'/calib.txt'
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # load LiDAR scans
    scan_folder = './semantic_kitti/dataset/sequences/'+seq_idx+'/velodyne'
    scan_paths = load_files(scan_folder)

    # test for the first N scans
    if num_frames >= len(poses) or num_frames <= 0:
        print('generate training data for all frames with number of: ', len(poses))
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    # generate residual images for the whole sequence
    residual_images = np.full((height, width, num_last_n), 0,
                          dtype=np.float32)  # [H,W] index (-1 is no data)

    # load current scan and generate current range image
    current_pose = poses[frame_idx]
    current_scan = load_vertex(scan_paths[frame_idx])
    current_range, proj_xyz, proj_intensity, proj_mask, proj_x_orig, proj_y_orig, current_vertex = range_projection(current_scan.astype(np.float32),
                                     height, width, fov_up, fov_down, max_range, min_range)

    # for the first N frame we generate a dummy file
    for i_idx in range(num_last_n):
        diff_image = np.full((height, width), 0,
                             dtype=np.float32)  # [H,W] range (0 is no data)
        if frame_idx < i_idx + 1:
            residual_images[:, :, i_idx] = diff_image
        else:
            # load last scan, transform into the current coord and generate a transformed last range image
            last_pose = poses[frame_idx - i_idx - 1]
            last_scan = load_vertex(scan_paths[frame_idx - i_idx - 1])
            last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
            last_range_transformed, _, _, _, _, _, _ = range_projection(last_scan_transformed.astype(np.float32),
                                                      height, width, fov_up, fov_down, max_range, min_range)

            # generate residual image
            valid_mask = (current_range > min_range) & \
                         (current_range < max_range) & \
                         (last_range_transformed > min_range) & \
                         (last_range_transformed < max_range)
            difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask])

            if normalize:
                difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / current_range[
                    valid_mask]

            diff_image[valid_mask] = difference

            if debug:
                fig, axs = plt.subplots(3)
                axs[0].imshow(last_range_transformed)
                axs[1].imshow(current_range)
                axs[2].imshow(diff_image, vmin=0, vmax=10)
                plt.show()

            residual_images[:, :, i_idx] = diff_image

    unproj_n_points = current_vertex.shape[0]
    # unproj_xyz = torch.full((150000, 3), -1.0, dtype=torch.float)
    # unproj_xyz[:unproj_n_points] = torch.from_numpy(points)
    # unproj_range = torch.full([150000], -1.0, dtype=torch.float)
    # unproj_range[:unproj_n_points] = torch.from_numpy(range)
    # unproj_remissions = torch.full([150000], -1.0, dtype=torch.float)
    # unproj_remissions[:unproj_n_points] = torch.from_numpy(remissions)

    proj_x = torch.full([150000], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(proj_x_orig)
    proj_y = torch.full([150000], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(proj_y_orig)

    proj_range = torch.from_numpy(current_range).clone()
    proj_xyz = torch.from_numpy(proj_xyz).clone()
    residual_images = torch.from_numpy(residual_images).clone()
    proj_intensity = torch.from_numpy(proj_intensity).clone()
    proj_mask = torch.from_numpy(proj_mask)

    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_intensity.unsqueeze(0).clone(),
                      residual_images.permute(2, 0, 1).clone()])

    proj_full = proj * proj_mask.float()
    proj_full = torch.cat([proj_full, proj_mask.unsqueeze(0).float()])

    return proj_full, proj_x, proj_y


# for debug
if __name__ == '__main__':
    seq_idx = '08'
    frame_idx = 15
    num_last_n = 3
    range_residual_images, _, _ = gen_range_residual_images(seq_idx, frame_idx, num_last_n)
    print(range_residual_images.size())

