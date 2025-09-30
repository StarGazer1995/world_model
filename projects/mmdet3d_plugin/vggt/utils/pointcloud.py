import numpy as np

def read_pcd_bin(file_path, dataset='default'):
    """Read .pcd.bin point cloud file
    
    Args:
        file_path (str): Path to .pcd.bin file
        dataset (str): Dataset format ('default' or 'nuscenes')
                      - default: x,y,z,intensity (4D)
                      - nuscenes: x,y,z,intensity,ring_index (5D)
    
    Returns:
        np.ndarray: N x point_dim array of point cloud data
    """
    # Read binary data
    data = np.fromfile(file_path, dtype=np.float32)
    
    # Reshape based on dataset format
    if dataset == 'nuscenes':
        points = data.reshape(-1, 5)  # x,y,z,intensity,ring_index
    else:
        points = data.reshape(-1, 4)  # x,y,z,intensity
    
    return points

# Example usage:
# Example usage:
# points = read_pcd_bin("test/lidar/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243097718.pcd.bin", dataset='nuscenes')
# print(points.shape)  # Should show (N, 5) for nuScenes
# print(points[:5])    # Show first 5 points with x,y,z,intensity,ring_index

if __name__ == "__main__":
    pcd = read_pcd_bin("test/lidar/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243097718.pcd.bin", dataset="nuscenes")
    print(pcd[:10])
