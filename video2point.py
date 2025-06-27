import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

if __name__ == '__main__':
    
    # Extract images from videos
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("path", type=str, help="input path to the video")
    parser.add_argument('valid_cam', type=str, help='valid camera numbers')
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
    images_path = os.path.join(save_dir, "images/") # data_sparse/N3V/coffee_martini/images/
    os.makedirs(images_path, exist_ok=True)
    assert os.path.exists(save_dir), f"save_dir {save_dir} does not exist"
    
    for video in videos:
        cam_name = video.split('/')[-1].split('.')[-2]
        do_system(f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png")
    print(f"Extracted images from {len(videos)} videos to {images_path}")
        
    # load data
    images = [f[len(save_dir):] for f in sorted(glob.glob(os.path.join(images_path, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
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
    image_names = [os.path.join(images_path, img)
                for img in sorted(os.listdir(images_path)) # data_sparse/N3V/coffee_martini/images/
                if img.endswith('0000.png') ] # only use the first frame of each video

    assert len(image_names) == n_video, "Number of images should match number of videos"
    
    original_sizes = []
    for image_path in image_names:
        img = Image.open(image_path)
        original_sizes.append(img.size)
        img.close()
    original_sizes = set(original_sizes)
    assert len(original_sizes) == 1, "Images should have the same size"
    
    images_tensor = load_and_preprocess_images(image_names).to(device) # 注意这一行代码会改变原始图像大小
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
    poses_bounds = np.zeros((N, 17), dtype=np.float32)  # 初始化 [N, 17]
    
    for i in range(len(intrinsic)):
        intrinsic[i, 0, 0] *= original_W / W
        intrinsic[i, 1, 1] *= original_H / H
        intrinsic[i, 0, 2] *= original_W / W
        intrinsic[i, 1, 2] *= original_H / H

    for i in range(N):
        # 提取外参 [3, 4]
        pose = extrinsic[i, :3, :4].cpu().numpy()  # [3, 4]
        
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
        
        # 估计深度范围
        near = depth_map[i].min().cpu().numpy()
        far = depth_map[i].max().cpu().numpy()
        poses_bounds[i, -2:] = [near, far]
    
    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses_bounds as {poses_bounds.shape}')

    assert N == len(cams)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    bounds = poses_bounds[:, -2:] # (N, 2)

    H, W, fl = poses[0, :, -1] 

    print(f'[INFO] H = {H}, W = {W}, fl = {fl}')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

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
        'w': int(W),
        'h': int(H),
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': int(cx),
        'cy': int(cy),
        'frames': train_frames,
    }
    test_transforms = {
        'w': int(W),
        'h': int(H),
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': int(cx),
        'cy': int(cy),
        'frames': test_frames,
    }

    train_output_path = os.path.join(save_dir, 'transforms_train.json')
    test_output_path = os.path.join(save_dir, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    # Extract points and colors
    points = point_map_by_unprojection.reshape(-1, 3)  # [S*H*W, 3]

    confidences = depth_conf.squeeze(0).view(-1).cpu().numpy()  # [S*H*W]
    images_rgb = images_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # [S, H, W, 3]
    colors = images_rgb.reshape(-1, 3)  # [S*H*W, 3]

    # Filter points by confidence
    threshold = 0.99
    valid = confidences > threshold
    points = points[valid]
    colors = colors[valid]  # Scale to [0, 255] for uint8
    colors = np.clip(colors, 0.0, 1.0)  # Clip to [0, 255] and convert to uint8

    # Save colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    output_ply = os.path.join(save_dir, f"points3d.ply")
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Colored point cloud saved to {output_ply}")

    