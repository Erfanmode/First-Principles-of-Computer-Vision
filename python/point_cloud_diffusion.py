# python/point_cloud_diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt

from pointnet_model import PointCloudNoisePredictor
from visualization import plot_training_loss, plot_point_cloud_3d # Assuming these functions are available


# --- 0. Configuration Parameters ---
class Config:
    N_POINTS = 2048       # Number of points per point cloud (target sampled size)
    POINT_DIM = 3          # Dimension of points (x, y, z)
    EMBED_DIM = 128        # Feature embedding dimension for PointNet layers
    T_STEPS = 1000         # Total number of diffusion time steps (T)
    
    # TODO: You can modify these training parameters as needed
    BATCH_SIZE = 32        # Batch size for training (can be modified)
    N_EPOCHS = 20        # Training Epochs (can be modified)
    LEARNING_RATE = 1e-3   # Learning Rate (can be modified)
    VAL_STEP = 50          # Perform a validation/checkpoint save every N batches
    
    # Checkpoint file path templates
    CHECKPOINT_PATH_EPOCH = '../results/model_epoch_final.pth' # Path to save model weights after final epoch
    CHECKPOINT_PATH_STEP_TEMPLATE = '../results/model_step_{step:06d}.pth' # Template for intermediate step checkpoints

config = Config()


# --- 1. Data Loading Utility (Includes Random Sampling Logic) ---
def load_shapenet_split(split_name='train', base_dir='../data/03001627'):
    """
    Loads point cloud data for a ShapeNet 03001627 subset and uniformly samples the point count.
    
    Args:
        split_name (str): 'train', 'val', or 'test'.
        base_dir (str): Base directory containing the ShapeNet data split folders.
        
    Returns:
        torch.Tensor: Tensor containing all sampled point clouds (Total_Samples, N_POINTS, 3).
    """
    target_n_points = config.N_POINTS 
    split_path = Path(base_dir) / split_name
    print(f"Attempting to load data from: {split_path}")
    
    if not split_path.is_dir():
        print(f"[ERROR] Directory not found: {split_path}")
        print("Please ensure the '03001627' directory is correctly placed inside '../data/'")
        return None

    point_clouds = []
    files = sorted([f for f in os.listdir(split_path) if f.endswith('.npy')])
    
    if not files:
        print(f"[WARNING] No .npy files found in {split_path}")
        return None

    for file in files:
        path = split_path / file
        try:
            data = np.load(path) # Shape: (N_original, 3)
            N_original = data.shape[0]
            
            # --- Random Sampling/Upsampling Logic to ensure N_POINTS ---
            if N_original != target_n_points:
                if N_original >= target_n_points:
                    # Downsampling (sample without replacement)
                    choice = np.random.choice(N_original, target_n_points, replace=False)
                else:
                    # Upsampling/Padding (sample with replacement)
                    choice = np.random.choice(N_original, target_n_points, replace=True)
                
                data = data[choice, :]

            # TODO: Do we need to normalize or center the point clouds here? If so, add that logic.
            data_mean = data.mean(axis=0, keepdims=True)
            data_std = data.std(axis=0, keepdims=True)
            normalized_data = (data - data_mean) / data_std

            point_clouds.append(torch.tensor(normalized_data, dtype=torch.float32))
            
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            continue
        
    if not point_clouds:
        return None

    stacked_data = torch.stack(point_clouds)
    print(f"[INFO] Data loaded and uniformly sampled/padded to shape: {stacked_data.shape[1:]}")
    
    return stacked_data


# --- 2. Diffusion Scheduler (Forward and Reverse Processes) ---
class DiffusionScheduler(nn.Module):
    def __init__(self, device, schedule_type="cosine", t_steps=1000):
        super().__init__()

        self.T = t_steps
        self.device = device
        # Register the noise schedule
        if schedule_type == "linear":
            betas = self.linear_beta_schedule(t_steps)
        elif schedule_type == "cosine":
            betas = self.cosine_beta_schedule(t_steps)
        else:
            raise ValueError("schedule_type must be 'linear' or 'cosine'")
        
        # (T,)
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)   # ᾱ1, ᾱ2, ..., ᾱT
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0],device=self.device), alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # For numerical stability in extreme cases
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    # Noise Schedules
    # -----------------------------
    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.05):
        return torch.linspace(start, end, timesteps, device=self.device)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.02)
    
    # Forward Process Sampling: q(x_t | x_0)
    def forward_sample(self, x0, t):
        """
        Samples x_t given x_0 and time step t using the forward diffusion process.
        N is number of points.
        Args:
            x0 (torch.Tensor): Original point cloud (B, N, 3).
            t (torch.Tensor): Time steps (B,).
        Returns:
            x_t (torch.Tensor): Noisy point cloud at time t (B, N, 3).
            epsilon (torch.Tensor): The noise added (B, N, 3).
        """
        # TODO: Implement the forward diffusion sampling logic here
        # t: (B,) → ensure long type for indexing
        t = t.long()

        # Sample noise: ε ~ N(0, I)
        epsilon = torch.randn_like(x0)  # (B, N, 3)

        # Extract √ᾱt and √(1 - ᾱt) for the given t (vectorized over batch)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]          # (B,)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t] # (B,)

        # Reshape for broadcasting: (B, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)

        # x_t = √ᾱt * x0 + √(1 - ᾱt) * ε
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_t * epsilon

        return x_t, epsilon

    # Reverse Process Step: Denoising from x_t to x_{t-1}
    @torch.no_grad()
    def reverse_step(self, xt, t, predicted_noise):
        """
        Performs one reverse diffusion step from x_{t} to x_{t-1}.
        N is the number of points.
        Args:
            xt (torch.Tensor): Noisy point cloud at time t (B, N, 3).
            t (torch.Tensor): Current time step. (B,)
            predicted_noise (torch.Tensor): Predicted noise by the model (B, N, 3).
        
        Returns:
            x_{t-1} (torch.Tensor): Denoised point cloud at time t-1 (B, N, 3).
        """
        # TODO: Implement the reverse diffusion step logic here
        # t: (B,) → ensure long type for indexing
        t = t.long()                  # (B,)
        beta_t = self.betas[t].view(-1, 1, 1)       # (B, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)        # (B, 1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1) # (B, 1, 1)

        # Compute the mean μ_θ(x_t, t)
        # μ = 1/sqrt(α_t) * (x_t - (β_t / sqrt(1 - ᾱ_t)) * ε_θ )
        mean = (1 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) # (B, N, 3)

        # Posterior variance σ_t² (Both cases are valid based on https://doi.org/10.48550/arXiv.2006.11239)
        # 1
        # alpha_bar_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)   # ᾱ_{t-1} (B, 1, 1)
        # posterior_var = beta_t* (1 - alpha_bar_prev) / (1 - alpha_bar_t) # (B, 1, 1)
        # 2
        posterior_var = beta_t # (B, 1, 1)

        # ratio of 0.8 causes better convergence of denoising process
        noise = torch.randn_like(xt)*0.8 # (B, N, 3) 
        xt_prev = mean + torch.sqrt(posterior_var) * noise # (B, N, 3)

        return xt_prev # (B, N, 3)

@torch.no_grad()
def generate(model, scheduler, device, num_samples=4):
    model.eval()
    
    # 1. Start generation from pure noise (x_T)
    xt = torch.randn(num_samples, config.N_POINTS, config.POINT_DIM, device=device) 
    
    print(f"\n--- Starting Generation (T={config.T_STEPS} steps) ---")
    
    # TODO: Implement the full generation loop here
    for t in reversed(range(config.T_STEPS)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Model predicts noise: ε̂_θ(x_t, t)
        predicted_noise = model(xt, t_tensor)

        # Scheduler computes x_{t-1}
        xt = scheduler.reverse_step(xt, t_tensor, predicted_noise)

        if t % max(config.T_STEPS // 10, 1) == 0:
            print(f"Step {t:03d} → {t-1:03d}")

    print("--- Generation Complete ---\n")
    return xt  # final denoised point clouds

def visualize_denoising_process(model, scheduler, device, num_steps=25, save_path='../results/denoising_process.png'):
    """
    Visualizes the denoising process of the diffusion model over a specified number of steps.
    Saves all snapshots in one figure.
    Args:
        model: The trained diffusion model.
        scheduler: The diffusion scheduler.
        device: The computation device (CPU/GPU).
        num_steps (int): Number of denoising steps to visualize.
        save_path (str): Path to save the visualization image.
    """
    # TODO: Implement the denoising process visualization logic here
    model.eval()

    # Start from random noise (single example)
    xt = torch.randn(1, config.N_POINTS, config.POINT_DIM, device=device)

    # Choose which timesteps to visualize (evenly spaced)
    timesteps = np.linspace(config.T_STEPS - 1, 0, num_steps, dtype=int)

    snapshots = []

    print(f"\n--- Visualizing Denoising Process ({num_steps} snapshots) ---")

    # Run full reverse diffusion loop, save snapshots when t matches
    for t in reversed(range(config.T_STEPS)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        predicted_noise = model(xt, t_tensor)
        xt = scheduler.reverse_step(xt, t_tensor, predicted_noise)

        if t in timesteps:
            snapshots.append(xt[0].detach().cpu().numpy())
            print(f"Stored snapshot at step t={t}")

    
    # Visualization grid
    cols = 5
    rows = int(np.ceil(num_steps / cols))

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    for i, pc in enumerate(snapshots):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.view_init(elev=-45, azim=25, roll=90)
        ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            s=2, alpha=0.8
        )

        ax.set_title(f"t={timesteps[i]}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nSaved denoising visualization to: {save_path}\n")

# --- 3. Training and Generation Functions ---
def train(model, scheduler, device, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epoch_losses = []
    global_step = 0
    
    print(f"--- Starting Training on {device} ---")
    start_time = time.time()
    
    for epoch in range(config.N_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            x0 = batch_data[0].to(device)  # Clean point clouds: (B, N, 3)
            B = x0.shape[0]

            # --- Training Step ---
            
            # Sample random timesteps t ~ Uniform({0, ..., T-1})
            t = torch.randint(0, config.T_STEPS, (B,), device=device).long()
            
            # TODO: Complete the training step logic below
            # Forward diffusion: add noise → get x_t and ground-truth ε
            x_t, epsilon = scheduler.forward_sample(x0, t)  

            # Predict noise with the model ε_θ(x_t, t)
            predicted_epsilon = model(x_t, t)

            # Simple noise prediction loss (L2)
            loss = F.mse_loss(predicted_epsilon, epsilon)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            # Optional: gradient clipping to stabilize learning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # --- Validation and Checkpoint Step ---
            if global_step % config.VAL_STEP == 0:
                model.eval()
                with torch.no_grad():
                    # Calculate Approximate Validation Loss using the current batch
                    val_loss = F.mse_loss(epsilon, predicted_epsilon).item()
                    
                    print(f"  [VAL Step {global_step}] Batch Loss: {val_loss:.6f}")
                    
                    # Save intermediate checkpoint using the step number
                    try:
                        step_path = config.CHECKPOINT_PATH_STEP_TEMPLATE.format(step=global_step)
                        torch.save(model.state_dict(), step_path)
                        print(f"  [CHECKPOINT] Model weights saved to {step_path}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to save step checkpoint: {e}")
                model.train() # Return to training mode
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.N_EPOCHS}] Average Loss: {avg_loss:.6f}")
        epoch_losses.append(avg_loss)

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # --- SAVE FINAL MODEL WEIGHTS ---
    try:
        # Save the final model's state dictionary
        torch.save(model.state_dict(), config.CHECKPOINT_PATH_EPOCH)
        print(f"[INFO] Final model weights saved to {config.CHECKPOINT_PATH_EPOCH}")
    except Exception as e:
        print(f"[ERROR] Failed to save final model weights: {e}")
    
    # --- Training Visualization (imported from evaluation.py) ---
    plot_training_loss(epoch_losses)


# --- 4. Main Program Entry Point ---
def main(do_train=True):
    # Ensure output directories exist
    os.makedirs('../results', exist_ok=True)
    # os.makedirs('python', exist_ok=True) 

    # Determine the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model and Scheduler
    model = PointCloudNoisePredictor(
        config.N_POINTS, 
        config.POINT_DIM, 
        config.EMBED_DIM, 
        config.T_STEPS
    ).to(device)
    scheduler = DiffusionScheduler(device, "linear", config.T_STEPS)
    
    if do_train :
        # Load data (includes sampling logic)
        all_data = load_shapenet_split('train')
        
        if all_data is None:
            print("[FATAL] Could not load training data. Exiting.")
            return
        print(f"Loaded {all_data.shape[0]} training samples with unified shape {all_data.shape[1:]}")
        
        # Convert to PyTorch DataLoader
        dataset = TensorDataset(all_data)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

        # Train model and save checkpoints
        train(model, scheduler, device, dataloader)

    else:
        model_path = config.CHECKPOINT_PATH_EPOCH
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate samples using the trained model
    generated_samples = generate(model, scheduler, device, num_samples=4)
    visualize_denoising_process(model, scheduler, device, num_steps=25, save_path='../results/denoising_process.png')

    # Save final generated point clouds
    generated_samples_np = generated_samples.detach().cpu().numpy()
    np.save('../results/generated_points.npy', generated_samples_np)
    print("\n[INFO] Generated samples saved to ../results/generated_points.npy")
    plot_point_cloud_3d(generated_samples_np, 4)


if __name__ == '__main__':
    main(do_train=False) # Change do_train to True for training