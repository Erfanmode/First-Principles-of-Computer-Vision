import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from nerf_dataset import NerfDataset
from nerf_model import NeRF

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Default tensor type set to torch.cuda.FloatTensor")
else:
    print("CUDA not available")


################################ IMPORTANT: This model is quite slow, you do not need to run it until it converges.  ###################################

# Position Encoding
class PositionalEncoder(nn.Module):
    """
    Implement the Position Encoding function.
    Defines a function that embeds x to (sin(2^k*pi*x), cos(2^k*pi*x), ...)
    Please note that the input tensor x should be normalized to the range [-1, 1].

    Args:
    x (torch.Tensor): The input tensor to be embedded.
    L (int): The number of levels to embed.

    Returns:
    torch.Tensor: The embedded tensor.
    """
    def __init__(self, data_range, L):
        super(PositionalEncoder, self).__init__()
        self.data_range = torch.tensor(data_range)
        self.scale = lambda x: (x - torch.mean(self.data_range, 0)) /\
            (self.data_range[1] - self.data_range[0])
        self.L = L
        # Final output dimension
        self.out_dim = 2 * L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (... , D) tensor, assumed to lie in [-data_range, data_range]
        
        Returns:
            encoded: (... , 2*L) tensor
        """
        # Normalize to [-1, 1] if needed
        x = self.scale(x)
        
        # Shape: (L,)
        frequencies = 2.0 ** torch.linspace(0, self.L - 1, self.L, device=x.device, dtype=x.dtype)
        # Shape: (L,)
        frequencies = frequencies * np.pi   # π * 2^k
        
        # Shape: (... , D, L)
        x_freq = x.unsqueeze(-1) * frequencies.unsqueeze(0)
        
        # Compute sin and cos
        sin_part = torch.sin(x_freq)   # (... , D, L)
        cos_part = torch.cos(x_freq)   # (... , D, L)
        
        # Concatenate along the feature dimension
        encoded = torch.cat([sin_part, cos_part], dim=-1)  # (... , D, 2*L)
        
        # Reshape to (... , 2*L) by flattening the input dimensions
        encoded = encoded.flatten(start_dim=-2, end_dim=-1) if encoded.ndim > 2 else encoded
        
        return encoded


def sample_rays(H, W, f, c2w):
    """
    Samples rays from a camera with given height H, width W, focal length f, and camera-to-world matrix c2w.

    Args:
    H (int): The height of the image.
    W (int): The width of the image.
    f (float): The focal length of the camera.
    c2w (torch.Tensor): The 4x4 camera-to-world transformation matrix.

    Returns:
    rays_o (torch.Tensor): The origin of each ray, with shape (W, H, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (W, H, 3).
    """
    # Handle both single matrix and batched
    batched = c2w.ndim == 3
    if not batched:
        c2w = c2w.unsqueeze(0)  # (1, 4, 4)

    # Create pixel grid: i = x (width), j = y (height)
    i, j = torch.meshgrid(
        torch.arange(W, device=c2w.device, dtype=torch.float32),
        torch.arange(H, device=c2w.device, dtype=torch.float32),
        indexing='xy'  # i correspond to x, j to y
    ) # (H, W)
    i = i.transpose(0, 1)  # (W, H)
    j = j.transpose(0, 1)  # (W, H)

    # Convert to normalized camera coordinates (centered at optical center)
    # Assuming principal point is at (W/2, H/2)
    dirs = torch.stack([
        (i - W * 0.5) / f,           # x: (u - cx) / f
        -(j - H * 0.5) / f,          # y: -(v - cy) / f   (negative because image y is down)
        -torch.ones_like(i)          # z: -1 (forward direction in camera space)
    ], dim=-1)  # Shape: (W, H, 3)

    # Rotate directions from camera space to world space
    # c2w[:, :3, :3] are the rotation matrices R (camera to world)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:, :3, :3], dim=-1)  # (B, W, H, 3)
    
    # Translate camera origin to world space
    rays_o = c2w[:, :3, -1].expand(rays_d.shape)  # (B, W, H, 3)

    # Normalize direction vectors (optional but common)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # Remove batch dimension if input wasn't batched
    if not batched:
        rays_o = rays_o.squeeze(0)
        rays_d = rays_d.squeeze(0)

    return rays_o, rays_d

def sample_points_along_the_ray(
    rays_o: torch.Tensor,      # (N_rays, 3) or (B, N_rays, 3)
    rays_d: torch.Tensor,      # (N_rays, 3) — must be normalized!
    tn: torch.Tensor,          # (N_rays, 1) or scalar — near distance
    tf: torch.Tensor,          # (N_rays, 1) or scalar — far distance
    N_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points along camera rays (NeRF-style stratified sampling).

    Args:
        rays_o:        Ray origins              [..., 3]
        rays_d:        Ray directions (normalized) [..., 3]
        tn:            Near bound per ray       [..., 1] or scalar
        tf:            Far bound per ray        [..., 1] or scalar
        N_samples:     Number of samples per ray
        perturb:       Add random jitter (stratified sampling)

    Returns:
        points:    3D coordinates      [..., N_samples, 3]
        t_vals:    Distances along ray [..., N_samples]
    """
    # Ensure tn/tf are tensors with correct shape
    if not torch.is_tensor(tn):
        tn = torch.full_like(rays_o[..., :1], float(tn))
    if not torch.is_tensor(tf):
        tf = torch.full_like(rays_o[..., :1], float(tf))

    # Shape for broadcasting
    batch_shape = rays_o.shape[:-1]

    # Uniform in depth 
    t_vals = torch.linspace(0, 1, N_samples, device=rays_o.device)
    t_vals = t_vals.expand(*batch_shape, N_samples)
    t_vals = tn + (tf - tn) * t_vals

    if perturb:
        noise = (torch.rand_like(t_vals) - 0.5) * (tf - tn) / N_samples
        t_vals = t_vals + noise

    # Final: compute 3D points = origin + t * direction
    points = rays_o[..., None, :] + t_vals[..., :, None] * rays_d[..., None, :]

    return points, t_vals
    

def volumn_render(NeRF, rays_o, rays_d, N_samples):
    """
    Performs volume rendering to generate an image from rays.

    Args:
    NeRF (nn.Module): The neural radiance field model.
    rays_o (torch.Tensor): The origin of each ray, with shape (N_rays, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (N_rays, 3).
    N_samples (int): The number of samples to take along each ray.

    Returns:
    torch.Tensor: The rendered RGB image.
    """
    # Sample points along each ray, from near plane to far plane
    # Calculate the points along the rays by sampling
    # pts.shape => (N_rays, N_samples, 3)
    points, t_vals = sample_points_along_the_ray(rays_o, rays_d, tn=2.0, tf=6.0, N_samples=N_samples)
    N_rays = t_vals.shape[0]
    if len(rays_d.shape)==2:
        rays_d_expanded = rays_d.unsqueeze(-2).expand_as(points)
    else:
        rays_d_expanded = rays_d
    
    # Distance between consecutive samples
    dists = torch.cat([
        t_vals[..., 1:] - t_vals[..., :-1],
        torch.full((N_rays, 1), 1e10)  # last interval = infinity
    ], dim=-1)  # (N_rays, N_samples)

    # Get the color and density from the NeRF model
    rgb, sigma = NeRF(points, rays_d_expanded)
    alpha = 1. - torch.exp(-dists* torch.nn.Softplus()(sigma.squeeze(-1)))
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=1)    # Transmittance
    T = torch.cat([torch.ones_like(T[:, :1]), T], dim=1)  # pad first T=1
    T = T[:, :-1]  # remove last (unused)

    # Volume rendering: compute the transmittance and accumulate the color
    # alpha = 1. - torch.exp(-delta * torch.relu(sigma)) # original formula in the paper
    # alpha = 1. - torch.exp(-delta * torch.nn.Softplus()(sigma)) # you can choose the trick to stabilize training
    
    # Compute the weights for each sample using alpha compositing
    weights = alpha * T              # (N_rays, N_samples)

    # 5. Composite color
    # Accumulate the color along each ray
    # using Trapezoidal Rule
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # (N_rays, 3)

    return rgb_map
    


def random_select_rays(H, W, rays_o, rays_d, img, N_rand):
    """
    Randomly select N_rand rays to reduce memory usage.
    Do not accept batch structure!!!

    Parameters:
    - H: int, height of the image.
    - W: int, width of the image.
    - rays_o: torch.Tensor, original ray origins with shape (H * W, 3).
    - rays_d: torch.Tensor, ray directions with shape (H * W, 3).
    - img: torch.Tensor, image with shape (H * W, 3).
    - N_rand: int, number of random rays to select.

    Returns:
    - selected_rays_o: torch.Tensor, selected ray origins with shape (N_rand, 3).
    - selected_rays_d: torch.Tensor, selected ray directions with shape (N_rand, 3).
    - selected_img: torch.Tensor, selected image pixels with shape (N_rand, 3).
    """
    # Generate coordinates for all pixels in the image
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    
    # Randomly select N_rand indices without replacement
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
    
    # Select the corresponding coordinates, rays, and image pixels
    select_coords = coords[select_inds].long().to("cpu")  # (N_rand, 2)
    selected_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = torch.tensor(selected_img, dtype=torch.float32)  # Ensure float32 dtype
    
    return selected_rays_o, selected_rays_d, selected_img


def fit_images_and_calculate_psnr(data_path, epochs=2000, learning_rate=5e-4):
    # get available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    dataset = NerfDataset(data_path)

    # create model
    xyz_encoder = PositionalEncoder(data_range=[-4., 4.], L=10).to(device)
    dir_encoder = PositionalEncoder(data_range=[-1., 1.], L=4).to(device)
    nerf = NeRF(
        xyz_encoder=xyz_encoder,
        dir_encoder=dir_encoder,
        input_dim=60,
        view_dim=24,
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    # train the model
    N_samples = 64 # number of samples per ray
    N_rand = 1024 # number of rays per iteration, adjust according to your GPU memory
    for epoch in tqdm(range(epochs+1)):
        for i in range(len(dataset)):
            img, pose, focal = dataset[i]
            img = img.to(device)
            H, W = img.shape[:2]
            pose = pose.to(device)
            focal = focal.to(device)

            # sample rays
            rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

            # random select N_rand rays to reduce memory usage
            selected_rays_o, selected_rays_d, selected_gt_rgb = random_select_rays(H, W, rays_o, rays_d, img, N_rand)

            # volumn render
            pred_rgb = volumn_render(NeRF=nerf, rays_o=selected_rays_o, rays_d=selected_rays_d, N_samples=N_samples)

            l = loss(pred_rgb, selected_gt_rgb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            psnr_value = psnr(selected_gt_rgb.detach().cpu().numpy(), pred_rgb.detach().cpu().numpy(), data_range=1)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {l.item()}, PSNR: {psnr_value}')
            with torch.no_grad():
                chunk_size = 1024 # adjust according to your GPU memory
                pred_rgb = []
                for i in range(0, H*W, chunk_size):
                    rays_o_chunk = rays_o.reshape(-1, 3)[i:i+chunk_size]
                    rays_d_chunk = rays_d.reshape(-1, 3)[i:i+chunk_size]
                    pred_rgb.append(volumn_render(NeRF=nerf, rays_o=rays_o_chunk, rays_d=rays_d_chunk, N_samples=N_samples))
                pred_rgb = torch.cat(pred_rgb, dim=0)
                torchvision.utils.save_image(pred_rgb.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'../output/NeRF/pred_{epoch}.png')
                torchvision.utils.save_image(img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'../output/NeRF/gt_{epoch}.png')

                
if __name__ == '__main__':
    data_path = '../data/lego' # data path
    psnr_value = fit_images_and_calculate_psnr(data_path)