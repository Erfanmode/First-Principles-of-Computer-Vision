# python/evaluation.py

import torch
import numpy as np

def compute_nearest_neighbor_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared distance from each point in pc1 to its nearest neighbor in pc2.
    
    Args:
        pc1 (torch.Tensor): Point cloud 1 (..., N1, D).
        pc2 (torch.Tensor): Point cloud 2 (..., N2, D).
        
    Returns:
        torch.Tensor: Squared distance from pc1 to pc2 (..., N1).
    """
    
    # Expand tensors to facilitate distance calculation across all pairs
    # pc1_expanded: (..., N1, 1, D)
    # pc2_expanded: (..., 1, N2, D)
    pc1_expanded = pc1.unsqueeze(-2)
    pc2_expanded = pc2.unsqueeze(-3)
    
    # Calculate squared Euclidean distance: ||pc1_i - pc2_j||^2
    # The result 'dist_sq': (..., N1, N2)
    # (..., N1, N2, D) -> (..., N1, N2) by summing over dimension D
    dist_sq = torch.sum((pc1_expanded - pc2_expanded) ** 2, dim=-1)
    
    # Find the minimum squared distance for each point in pc1 across pc2
    # min_dist_sq: (..., N1)
    min_dist_sq, _ = torch.min(dist_sq, dim=-1)
    
    return min_dist_sq


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Computes the Chamfer Distance between two batched point clouds.
    
    CD(P1, P2) = sum_{x in P1} min_{y in P2} ||x - y||^2 + sum_{y in P2} min_{x in P1} ||y - x||^2
    
    Args:
        pc1 (torch.Tensor): Point cloud 1 (..., B, N, D).
        pc2 (torch.Tensor): Point cloud 2 (..., B, N, D).
        reduce (bool): If True, returns the mean CD across the batch. (...,)
                       If False, returns the CD for each batch element (..., B).
        
    Returns:
        torch.Tensor: The Chamfer Distance.
    """
    # TODO: Implement the Chamfer Distance calculation using the nearest neighbor distances
    min_dist1 = compute_nearest_neighbor_distance(pc1, pc2) # (..., B, N)
    min_dist2 = compute_nearest_neighbor_distance(pc2, pc1) # (..., B, N)

    # Chamfer terms
    # cd1 = min_dist1.sum(dim=-1)   # (..., B)
    # cd2 = min_dist2.sum(dim=-1)   # (..., B)

    # Chamfer terms (Normalized when number of pointclouds are variant)
    cd1 = min_dist1.mean(dim=-1)   # (..., B)
    cd2 = min_dist2.mean(dim=-1)   # (..., B)

    chamfer_dist = cd1 + cd2      # (..., B)

    if reduce:
        return chamfer_dist.mean(dim=-1) # (...)
    else:
        return chamfer_dist # (..., B)

def minimum_matching_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Minimum Matching Distance (MMD) between two sets of point clouds.
    P1 -> generated data
    P2 -> ground-truth
    MMD(P1, P2) = (1/|P2|) * sum_{p2 in P2} min_{p1 in P1} CD(p1, p2)
    
    Args:
        pc1 (torch.Tensor): Set of point clouds 1 (B1, N, D).
        pc2 (torch.Tensor): Set of point clouds 2 (B2, N, D).
        
    Returns:
        torch.Tensor: The Minimum Matching Distance.
    """
    # TODO: Implement the Minimum Matching Distance calculation using Chamfer Distance
    B1 = pc1.shape[0]
    B2 = pc2.shape[0]
    device = pc1.device

    # Expand dimensions to compute all pairwise distances
    pc1_exp = pc1.unsqueeze(1)  # (B1, 1, N, D)
    pc2_exp = pc2.unsqueeze(0)  # (1, B2, N, D)

    # Compute Chamfer distannce
    # (slicing ground-truth based on available GPU memory)
    sum_min_CD = torch.tensor(0.0, device=device)
    ss = 1 # Slice Size (Larger the GPU memory, larger the slice size)
    N_iter = int(np.floor(B2/ss))
    for i in range(N_iter):
        Chamfer_dists = chamfer_distance(pc1_exp, pc2_exp[0:1, i:(i+1)*ss], reduce=False) # (B1, ss)
        min_CD, _ = torch.min(Chamfer_dists, dim=0) # (ss)
        sum_min_CD += torch.sum(min_CD)
        if i% 50 == 0:
            print(f"Progress: %{i/N_iter*100:.2f}")
            

    # Compute MMD
    MMD = sum_min_CD / B2

    return MMD
    


# --- Example Usage (Optional: For testing the metric) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    generated_path = '../results/generated_points.npy'
    generated_pc_np = np.load(generated_path)
    generated_pc = torch.tensor(generated_pc_np, dtype=torch.float32).to(device)
    
    # 2. Load ground truth point cloud
    from point_cloud_diffusion import load_shapenet_split
    gt_pc = load_shapenet_split("train")
    gt_pc = gt_pc.to(device)
    print(f"Loaded Generated PC shape: {generated_pc_np.shape}")
    print(f"Loaded Ground Truth PC shape: {gt_pc.shape}")

    # 3. Compute minimum matching distance 
    mmd = minimum_matching_distance(generated_pc, gt_pc)

    print(f"Minimum Matching Distance (MMD): {mmd.item()}")
    # Minimum Matching Distance (MMD): 0.1457
