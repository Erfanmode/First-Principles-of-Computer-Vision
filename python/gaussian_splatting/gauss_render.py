import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    '''
    Args:
    s (torch:Tensor): scale of gaussian with shape (N, 3)
    r (torch:Tensor): quaternion of gaussian with shape (N, 4)

    Using the provided build_rotation to get the rotation matrix from quaternion and build scale matrix for s.
    
    Return:
    (torch:Tensor) the matrix production of rotation matrix and scale matrix with shape (N, 3, 3) 
    '''
    R = build_rotation(r)
    return R * s.unsqueeze(1)     # (N, 3, 3) → R @ diag(s)


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def corvariance_3d(s, r):
    '''
    Args:
        s (torch:Tensor): scale of gaussian with shape (N, 3)
        r (torch:Tensor): quaternion of gaussian with shape (N, 4)
        
        We use build_scaling_rotation to build matrix of R S and then we can obtain the covariance of 3D gaussian by RS(RS)^T
    
    Return
        (torch:Tensor)) 3D covariance with shape (N, 3, 3)
    '''
    M = build_scaling_rotation(s, r)   # (N, 3, 3)
    cov = M @ M.transpose(1, 2)        # (N, 3, 3)
    return cov


def corvariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    """
    Compute the 2D image-space covariance for a batch of 3D Gaussians.

    Args:
        mean3d (torch.Tensor):     (N, 3)
            3D centers of Gaussians in world coordinates, one per Gaussian.

        cov3d (torch.Tensor):      (N, 3, 3)
            3D covariance matrices of Gaussians in world coordinates.

        viewmatrix (torch.Tensor): (4, 4)
            Camera extrinsic matrix (world-to-camera transform).
            The code uses viewmatrix[:3, :3] as rotation and viewmatrix[-1:, :3]
            as translation (note: last row stores translation in this convention).

        fov_x (float): horizontal field of view (radians).
        fov_y (float): vertical field of view (radians).
        focal_x (torch.Tensor): focal length in x, in pixel units.
        focal_y (torch.Tensor): focal length in y, in pixel units.

    Returns:
        cov2d (torch.Tensor): (N, 2, 2)
            2D covariance matrices in image (screen) space for each projected Gaussian,
            including a low-pass filter term for numerical stability / anti-aliasing.
    """
    viewmatrix=viewmatrix.T
    # Precompute tangent of half FOVs for frustum clipping in camera space and transform 3D Gaussian centers from world space to camera space.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    # TODO: project the 3d centers into camera space using mean3d and viewmatrix
    t = (viewmatrix[:3, :3] @ mean3d.T + viewmatrix[:3, 3:4]).T  # (N, 3)

    # Truncate Gaussians far outside the frustum.
    # We clip the normalized coordinates x/z, y/z, then re-scale by depth z.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # TODO: build the matrxi of J and cov2d
    J = torch.zeros((mean3d.shape[0], 3, 3), device=mean3d.device, dtype=mean3d.dtype)
    J[:, 0, 0] = focal_x / tz
    J[:, 1, 1] = focal_y / tz
    J[:, 0, 2] = -focal_x * tx / (tz * tz)
    J[:, 1, 2] = -focal_y * ty / (tz * tz)
    J[:, 2, 2] = 0  # z doesn't affect x/y after projection 

    R = viewmatrix[:3, :3]          # R_world→cam  
    cov3d_view = R @ cov3d @ R.transpose(-2, -1)

    # Project to 2D screen space
    cov2d = J @ cov3d_view @ J.transpose(-2, -1)
    cov2d = cov2d[:, :2, :2]          # (N, 2, 2)
    
    # add low pass filter here according to E.q. 32 of EWQ splatting
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d + filter.unsqueeze(0)


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= -10
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh

class GaussRenderer(nn.Module):

    def __init__(self, active_sh_degree=3, white_bkgd=True, pixel_range=256, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(pixel_range), torch.arange(pixel_range), indexing='xy'), dim=-1).to('cuda')
              
    
    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
    
    def render(self, camera, means2D, cov2d, color, opacity, depths):
        
        """
        Tile-based 2D Gaussian splatting renderer.

        Args:
            camera:
                Camera object providing image resolution:
                    - camera.image_width  (int)
                    - camera.image_height (int)
                self.pix_coord is assumed to be a precomputed (H, W, 2) tensor of pixel coords.

            means2D (torch.Tensor): (N, 2)
                Projected 2D centers of Gaussians in image space (pixel coordinates).

            cov2d (torch.Tensor): (N, 2, 2)
                2D image-space covariance matrices of Gaussians.

            color (torch.Tensor): (N, 3)
                RGB color for each Gaussian (in [0, 1]).

            opacity (torch.Tensor): (N, 1)
                Per-Gaussian base opacity (before per-pixel Gaussian weighting).

            depths (torch.Tensor): (N,)
                Per-Gaussian depth values (e.g., z in camera space),
                used for front-to-back sorting when compositing.
        Returns:
            dict with:
                - "render": (H, W, 3) final RGB image.
                - "alpha":  (H, W, 1) accumulated alpha map.
                - "visiility_filter": (N,) visibility mask (radii > 0).
                - "radii": (N,) per-Gaussian screen-space radius.
        """
        radii = get_radius(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        self.render_color = torch.zeros(*self.pix_coord.shape[:2], 3).to('cuda')
        """torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')"""
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        
        TILE_SIZE = 25
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue


                # TODO: Extract the pixel coordinates for this tile.
                # Hint: The tile's pixel coordinates should be extracted using slicing and flattening.
                tile_y, tile_x = torch.meshgrid(
                    torch.arange(h, min(h + TILE_SIZE, camera.image_height), device=means2D.device),
                    torch.arange(w, min(w + TILE_SIZE, camera.image_width), device=means2D.device),
                    indexing='ij'
                )
                tile_coord = torch.stack((tile_x, tile_y), dim=-1).reshape(-1, 2).float()  # (P, 2)
    
                # TODO: Sort Gaussians by depth.
                # Hint: Sorting should be based on the depth values of Gaussians.
                tile_depths = depths[in_mask]
                sorted_depths, index = torch.sort(tile_depths, descending=False)  # front-to-back
    
                # TODO: Extract relevant Gaussian properties for the tile.
                # Hint: Use the computed index to rearrange the following tensors.
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d   = cov2d[in_mask][index]
                sorted_opacity = opacity[in_mask][index].squeeze(-1)
                sorted_color   = color[in_mask][index]

                det = sorted_cov2d[:, 0, 0] * sorted_cov2d[:, 1, 1] - sorted_cov2d[:, 0, 1] * sorted_cov2d[:, 0, 1]
                det_inv = 1.0 / (det + 1e-6)
                sorted_conic = torch.stack([
                    sorted_cov2d[:, 1, 1] * det_inv,           # a
                    -sorted_cov2d[:, 0, 1] * det_inv,          # b
                    sorted_cov2d[:, 0, 0] * det_inv            # c
                    ], dim=1)  # (Ng, 3)

    
                # TODO: Compute the distance from each pixel in the tile to the Gaussian centers.
                # Hint: This involves computing dx between pixel coordinates and Gaussian centers.
                # You may need to use broadcasting: dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B N 2
                dx = tile_coord[:, None, :] - sorted_means2D[None, :, :]  # (P, Ng, 2)
    
                # TODO: Compute the 2D Gaussian weight for each pixel.
                # Hint: The weight is determined by the Mahalanobis distance using the covariance matrix.
                dx_x = dx[..., 0]
                dx_y = dx[..., 1]
                mah2 = (sorted_conic[None, :, 0] * dx_x * dx_x +
                        sorted_conic[None, :, 1] * dx_x * dx_y * 2 +
                        sorted_conic[None, :, 2] * dx_y * dx_y)
                gauss_weight = torch.exp(-0.5 * mah2.clamp(-50,50))  # (P, Ng)
    
                # TODO: Compute the alpha blending using transmittance (T).
                alpha = gauss_weight * sorted_opacity.clamp(0.0,0.99)  # (P, Ng)
                T_init = torch.ones(alpha.shape[0], 1, device=alpha.device)
                T = torch.cumprod(torch.cat([T_init, 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
                weight = alpha * T  # (P, Ng)
    
                # TODO: Compute the color and depth contributions.
                # Hint: Perform weighted summation using computed transmittance and opacity.
                tile_color = (weight.unsqueeze(-1) * sorted_color).sum(dim=1)  # (P, 3)
                tile_alpha = weight.sum(dim=1, keepdim=True)                  # (P, 1)

                # TODO: Store computed values into rendering buffers.
                # Hint: Assign tile-wise computed values to corresponding locations in the full image buffers.
                tile_h = tile_y.shape[0]
                tile_w = tile_x.shape[1]
                self.render_color[h:h+tile_h, w:w+tile_w] += tile_color.reshape(tile_h, tile_w, 3)
                self.render_alpha[h:h+tile_h, w:w+tile_w] += tile_alpha.reshape(tile_h, tile_w, 1)


        return {
            "render": self.render_color,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }


    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        

        mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                viewmatrix=camera.world_view_transform, 
                projmatrix=camera.projection_matrix)
        mean_ndc = mean_ndc  # [in_mask]
        mean_view = mean_view # [in_mask]
        depths = mean_view[:,2]
        
        color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        
        cov3d = corvariance_3d(scales, rotations)
            
        cov2d = corvariance_2d(
            mean3d=means3D, 
            cov3d=cov3d, 
            viewmatrix=camera.world_view_transform,
            fov_x=camera.FoVx, 
            fov_y=camera.FoVy, 
            focal_x=camera.focal_x, 
            focal_y=camera.focal_y)

        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
    
        rets = self.render(
            camera = camera, 
            means2D=means2D,
            cov2d=cov2d,
            color=color,
            opacity=opacity, 
            depths=depths,
        )

        return rets
