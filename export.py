import os
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
import torch
import numpy as np
from tqdm import tqdm
import numpy as np
import open3d as o3d



def unproject_depth_map(camera, depth_map):
    height, width = depth_map.shape
    x = torch.linspace(0, width - 1, width).cuda()
    y = torch.linspace(0, height - 1, height).cuda()
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Reshape the depth map and grid to N x 1
    depth_flat = depth_map.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    # Normalize pixel coordinates to [-1, 1]
    X_norm = (X_flat / (width - 1)) * 2 - 1
    Y_norm = (Y_flat / (height - 1)) * 2 - 1

    # Create homogeneous coordinates in the camera space
    points_camera = torch.stack([X_norm, Y_norm, depth_flat], dim=-1)
    # points_camera = points_camera.view((height,width,3))
    

    K_matrix = camera.projection_matrix
    # parse out f1, f2 from K_matrix
    f1 = K_matrix[2, 2]
    f2 = K_matrix[3, 2]
    # get the scaled depth
    sdepth = (f1 * points_camera[..., 2:3] + f2) / points_camera[..., 2:3]
    # concatenate xy + scaled depth
    points_camera = torch.cat((points_camera[..., 0:2], sdepth), dim=-1)


    points_camera = points_camera.view((height,width,3))
    points_camera = torch.cat([points_camera, torch.ones_like(points_camera[:, :, :1])], dim=-1)  
    points_world = torch.matmul(points_camera, camera.full_proj_transform.inverse())

    # Discard the homogeneous coordinate
    points_world = points_world[:, :, :3] / points_world[:, :, 3:]
    
    return points_world


def compute_normal_map(depth_map):
    # Assuming depth_map is a PyTorch tensor of shape [1, Height, Width]
    depth_map = depth_map[None, ...]
    # Define sobel filters for gradient computation in x and y direction
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
    

    if depth_map.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()

    # Compute gradients in x and y direction
    grad_x = torch.nn.functional.conv2d(depth_map, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(depth_map, sobel_y, padding=1)

    # Compute normal map
    dz = torch.ones_like(grad_x)
    normal_map = torch.cat((grad_x, grad_y, -dz), 0)
    norm = torch.norm(normal_map, p=2, dim=0, keepdim=True)
    normal_map = normal_map / norm

    return normal_map

def export(dataset, pipe, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pcd = o3d.geometry.PointCloud()
    points = torch.empty((0,3)).cuda()
    colors = torch.empty((0,3)).cuda()
    normals = torch.empty((0,3)).cuda()
    print("\nComputing Poisson Mesh")
    with torch.no_grad():
        for i in tqdm(range(len(cameras))):
            render_pkg = render(cameras[i], gaussians, pipe, background)
            image, depth = render_pkg["render"].permute(1,2,0), render_pkg["depth"].squeeze()
            normal = compute_normal_map(depth).permute(1,2,0)
            normal = torch.matmul(normal, cameras[i].world_view_transform[:3, :3].T)

            mask = depth!=15

            points = torch.cat((points, unproject_depth_map(cameras[i], depth)[mask]), dim=0)
            colors = torch.cat((colors, image[mask]), dim=0)
            normals = torch.cat((normals, normal[mask]), dim=0)
            
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print("\nSaving Poisson Mesh")
    o3d.io.write_triangle_mesh(os.path.join(args.model_path, 'mesh.ply'), mesh)
    
    
    


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=7000)
    args = parser.parse_args(sys.argv[1:])  
    print("Export " + args.model_path)

    export(lp.extract(args), pp.extract(args), args.iteration)

    print("\nExporting complete.")