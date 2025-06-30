import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

import open3d as o3d
import numpy as np

import argparse
import glob
from PIL import Image
import json
import sys

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def compute_camera_to_world_transforms(world_to_cam_matrices):
    """
    Compute inverse transforms (camera to world) from world-to-camera transformation matrices.
    
    Args:
        world_to_cam_matrices: numpy array of shape (N, 4, 4), where each 4x4 matrix
                              contains a 3x3 rotation matrix R and 3x1 translation vector T.
    
    Returns:
        cam_to_world_matrices: numpy array of shape (N, 4, 4), containing the inverse transforms.
    """
    # Ensure input is a numpy array with shape (N, 4, 4)
    world_to_cam_matrices = np.array(world_to_cam_matrices)
    if world_to_cam_matrices.shape[1:] != (4, 4):
        raise ValueError("Input matrices must have shape (N, 4, 4)")
    
    N = world_to_cam_matrices.shape[0]
    # Initialize output array for inverse transforms
    cam_to_world_matrices = np.zeros_like(world_to_cam_matrices)
    
    for i in range(N):
        # Extract 3x3 rotation matrix R and 3x1 translation vector T
        R = world_to_cam_matrices[i, :3, :3]
        T = world_to_cam_matrices[i, :3, 3]
        
        # Compute inverse: R^T and -R^T * T
        R_inv = R.T  # Inverse of rotation matrix is its transpose
        T_inv = -R_inv @ T  # Compute -R^T * T
        
        # Construct the 4x4 inverse transformation matrix
        cam_to_world_matrices[i, :3, :3] = R_inv
        cam_to_world_matrices[i, :3, 3] = T_inv
        cam_to_world_matrices[i, 3, :3] = 0  # Bottom row (0, 0, 0)
        cam_to_world_matrices[i, 3, 3] = 1   # Bottom right element is 1
    
    return cam_to_world_matrices

if __name__ == '__main__':
    
    # Extract images from videos
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("path", type=str, help="input path to the video")
    parser.add_argument('--valid_cam', type=str, help='valid camera numbers', default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    args = parser.parse_args()
    
    valid_cam = [int(c) for c in args.valid_cam.split(',')]
    
    # args.path 是 data 目录，例如：/data2/yangzf/4d-gaussian-splatting/data/coffee_martini
    # path must end with / to make sure image path is relative
    if args.path[-1] != '/':
        args.path += '/'
        
    # extract images
    videos = [os.path.join(args.path, vname) 
              for vname in os.listdir(args.path) 
              if vname.endswith(".mp4")
              and int(vname.split('.')[0][-2:]) in valid_cam]
    
    # save dir
    save_dir = args.path.replace('data/N3V', 'data_sparse/N3V') # data_sparse/N3V/coffee_martini
    assert os.path.exists(save_dir), f"save_dir {save_dir} does not exist"
        
    # load data
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    cams = sorted(set([im[7:12] for im in images]))
    
    # Generate point cloud and json file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT()
    model.load_state_dict(torch.load("checkpoint/model.pt"))
    model.to(device)
    print("Model loaded.")

    # Load and preprocess example images (replace with your own image paths)
    # 视频文件只在 data 有，照片文件在 data_sparse
    n_video = len(videos)
    assert n_video > 15, "No video found"
    image_names = [os.path.join(args.path, 'images', img)
                for img in sorted(os.listdir(os.path.join(args.path, 'images'))) # data/N3V/coffee_martini/images/
                if img.endswith('0000.png') ] # only use the first frame of each video
    
    assert len(image_names) == n_video, "Number of images should match number of videos"
    
    original_sizes = []
    for image_path in image_names:
        img = Image.open(image_path)
        original_sizes.append(img.size)
        img.close()
    original_sizes = set(original_sizes)
    assert len(original_sizes) == 1, "Images should have the same size"
    
    images_tensor = load_and_preprocess_images(image_names).to(device) 
    images_tensor = torch.clamp(images_tensor, 0.0, 1.0)
    print(f"Images loaded and preprocessed. Shape: {images_tensor.shape}") 
    
    # Generate 3D Point Cloud
            
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_tensor = images_tensor[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images_tensor)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_tensor, ps_idx)

        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images_tensor, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        extrinsic = extrinsic.squeeze(0)
        intrinsic = intrinsic.squeeze(0)
        depth_map = depth_map.squeeze(0)
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map, 
                                                                    extrinsic, 
                                                                    intrinsic)

    original_W, original_H = original_sizes.pop()
    print(f"Original image size: {original_W} x {original_H}")
    
    # 构造 poses_bounds.npy
    images_tensor = images_tensor.squeeze(0)
    N = images_tensor.shape[0]  # 相机数量
    H, W = images_tensor.shape[-2:]  # 图像高度和宽度
    poses_bounds = np.zeros((N, 15), dtype=np.float32)  # 初始化 [N, 15]
    
    for i in range(len(intrinsic)):
        intrinsic[i, 0, 0] *= original_W / W
        intrinsic[i, 1, 1] *= original_H / H
        intrinsic[i, 0, 2] *= original_W / W
        intrinsic[i, 1, 2] *= original_H / H

    for i in range(N):
        # 提取外参 [3, 4]
        pose = extrinsic[i, :3, :4].cpu().numpy()  # [3, 4] # 旋转矩阵拼上平移矩阵
        
        # 提取内参中的焦距
        fx = intrinsic[i, 0, 0].cpu().numpy() 
        fy = intrinsic[i, 1, 1].cpu().numpy()
        cx = intrinsic[i, 0, 2].cpu().numpy()
        cy = intrinsic[i, 1, 2].cpu().numpy()
        fl = (fx + fy) / 2.0
        
        # 构造 [3, 5] 位姿矩阵
        pose_with_hwfl = np.zeros((3, 5), dtype=np.float32)
        pose_with_hwfl[:, :4] = pose
        pose_with_hwfl[:, 4] = [original_H, original_W, fl]
        
        # 填充 poses_bounds 前 15 列
        poses_bounds[i, :15] = pose_with_hwfl.reshape(-1)
    
    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses_bounds as {poses_bounds.shape}')

    assert N == len(cams)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5) 

    print(f'[INFO] H = {H}, W = {W}, fx = {fx}, fy = {fy}, cx = {cx}, cy = {cy}, fl = {fl}')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    # poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)
    poses = np.concatenate([poses[..., 0:1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)

    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 
    
    poses = compute_camera_to_world_transforms(poses) # 将世界到相机转换为相机到世界
    
    # test poses

    # the following stuff are from colmap2nerf... 
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    poses[:, 2, :] *= -1 # flip whole world upside down

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        mf = poses[i, :3, :]
        for j in range(i + 1, N):
            mg = poses[j, :3, :]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            #print(i, j, p, w)
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp

    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()

    poses[:, :3, 3] *= 4.0 / avglen

    print(f'[INFO] average radius = {avglen}')
    
    train_frames = []
    test_frames = []
    for i in range(N):
        cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
                    'transform_matrix': poses[i].tolist(),
                    'time': int(im.lstrip("/").split('.')[0][-4:]) / 30.} for im in images if cams[i] in im]
        if i == 0:
            test_frames += cam_frames
        else:
            train_frames += cam_frames

    train_transforms = {
        'w': float(original_W),
        'h': float(original_H),
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'frames': train_frames,
    }
    test_transforms = {
        'w': float(original_W),
        'h': float(original_H),
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'frames': test_frames,
    }
    
    train_output_path = os.path.join(save_dir, f'transforms_train.json')
    test_output_path = os.path.join(save_dir, f'transforms_test.json')
    
    if os.path.exists(train_output_path) and os.path.exists(test_output_path) and len(valid_cam) == 21:
        # 生成用于渲染的 transforms_train.json 和 transforms_test.json
        file_paths = [train_output_path, test_output_path]
        for file_path in file_paths:
            new_file_path = file_path.replace('.json', '_sparse.json')
            os.replace(file_path, new_file_path)
            print(f'[INFO] rename {file_path} to {new_file_path}')
                
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
