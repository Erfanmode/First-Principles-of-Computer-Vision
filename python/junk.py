import torch
from train_3d_nerf import PositionalEncoder, sample_rays, sample_points_along_the_ray,\
      random_select_rays, volumn_render
from nerf_dataset import NerfDataset
from nerf_model import NeRF
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Default tensor type set to torch.cuda.FloatTensor")
else:
    print("CUDA not available")

dataset = NerfDataset("../data/lego","test")
img = dataset.imgs[21]
H, W = dataset.H, dataset.W
focal = dataset.focal
c2w = dataset.poses[21]
rays_o, rays_d = sample_rays(H, W, focal, c2w)
xyz_encoder = PositionalEncoder([-5.,5.], 10)
dir_encoder = PositionalEncoder([-1.,1.], 4)
nerf = NeRF(xyz_encoder, dir_encoder, 3*2*10, view_dim=3*2*4)
rendered_img = volumn_render(nerf, rays_o, rays_d, 64)
image_np = rendered_img.detach().cpu().numpy()
plt.imshow(image_np)
plt.axis('off') 
plt.show()
print("done")