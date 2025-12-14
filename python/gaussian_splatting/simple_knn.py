import torch
import torch.nn.functional as F

def average_nearest_neighbor_distance_3d(points, k=1):
    """
    Compute the average distance to the k-nearest neighbors for a set of 3D points.
    
    Args:
        points (torch.Tensor): Input points of shape (N, 3) where N is the number of points
        k (int): Number of nearest neighbors to consider (default: 1)
    
    Returns:
        torch.Tensor: Average distance to k-nearest neighbors for each point, shape (N,)
    """
    # Ensure inputs are CUDA tensors
    if not points.is_cuda:
        raise ValueError("Input points must be a CUDA tensor")
    
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    
    N = points.shape[0]
    
    # Compute pairwise distances using broadcasting
    # points: (N, 3), points.unsqueeze(1): (N, 1, 3), points.unsqueeze(0): (1, N, 3)
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 3)
    distances = torch.norm(diff, dim=2, p=2)  # (N, N) Euclidean distance
    
    # Set diagonal to a large value to exclude self-distances
    mask = torch.eye(N, device=points.device, dtype=torch.bool)
    distances = distances.masked_fill(mask, float('inf'))
    
    # Find k-nearest neighbors
    knn_distances, _ = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)
    
    # Compute average distance to k-nearest neighbors
    avg_distances = torch.mean(knn_distances, dim=1)
    
    return avg_distances
