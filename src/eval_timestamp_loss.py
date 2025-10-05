import os
from pathlib import Path
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings
from low_rank_compression import low_rank_layer_replacement, TimestepConditionedWrapper
import torch
from tqdm import tqdm
import torch.nn.functional as F
from preprocessing import create_dataloader
from config import TrainingConfig
import matplotlib.pyplot as plt
 



def eval_timestamp_loss(model, noise_scheduler, dataloader, device, max_batches_per_timestep=None):
    loss_dict = {}
    
    model.eval()
    
    # If limiting batches, pre-materialize a fixed list of batches to reuse each timestep
    fixed_batches = None
    if max_batches_per_timestep is not None:
        fixed_batches = []
        for step, batch in enumerate(dataloader):
            fixed_batches.append(batch)
            if step + 1 >= max_batches_per_timestep:
                break
    
    # Outer progress bar for timesteps
    timestep_pbar = tqdm(
        total=noise_scheduler.config.num_train_timesteps,
        desc="Evaluating timesteps",
        disable="SLURM_JOB_ID" in os.environ,
        position=0
    )
    
    for timestep in range(noise_scheduler.config.num_train_timesteps):
        total_loss = 0.0
        loss_dict[timestep] = 0.0
        
        # Determine how many batches to process for this timestep
        if fixed_batches is not None:
            batch_total = len(fixed_batches)
            batch_iterable = fixed_batches
        else:
            batch_total = len(dataloader)
            batch_iterable = dataloader

        # Inner progress bar for batches within current timestep
        batch_pbar = tqdm(
            total=batch_total,
            desc=f"Timestep {timestep:4d}",
            disable="SLURM_JOB_ID" in os.environ,
            leave=False,
            position=1
        )
        
        processed_batches = 0
        for step, batch in enumerate(batch_iterable):
            clean_images = batch["img"].to(device)
            noise = torch.randn(clean_images.shape, device=device)
            batch_size = clean_images.shape[0]

            timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

            class_labels = None
            if "label" in batch:
                class_labels = batch["label"].to(device)
            else:
                class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with torch.no_grad():
                noise_predicted = model(noisy_images, timesteps, class_labels, return_dict=False)[0]

            loss = F.mse_loss(noise_predicted, noise, reduction="none")
            loss = loss.mean()
            total_loss += loss.item()
            processed_batches += 1
            
            # Update batch progress bar with current loss info
            batch_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(processed_batches):.4f}'
            })
            batch_pbar.update(1)

            # If we've reached the maximum number of batches for this timestep, stop early
            if fixed_batches is None and max_batches_per_timestep is not None and processed_batches >= max_batches_per_timestep:
                break

        loss_dict[timestep] = total_loss / processed_batches if processed_batches > 0 else float('nan')
        
        # Update timestep progress bar with timestep loss
        timestep_pbar.set_postfix({
            'timestep_loss': f'{loss_dict[timestep]:.4f}'
        })
        timestep_pbar.update(1)
        
        batch_pbar.close()
    
    timestep_pbar.close()
    return loss_dict




def plot_loss_dict(loss_dict, title="Loss vs Timestep", xlabel="Timestep", ylabel="Avg Loss", save_path=None):
    """
    Plot a dictionary of losses vs timesteps as a line plot.
    
    Args:
        loss_dict (dict): Dictionary with timesteps as keys and loss values as values
        title (str): Title for the plot
        xlabel (str): Label for x-axis  
        ylabel (str): Label for y-axis
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    # Extract timesteps and losses
    timesteps = sorted(loss_dict.keys())
    losses = [loss_dict[t] for t in timesteps]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250903_110459" / "model_0599.pt"  

    config = TrainingConfig()
    model = create_model(config)
    # config.low_rank_pretraining = True

    if config.low_rank_pretraining:
        model = low_rank_layer_replacement(model, percentage=config.low_rank_rank, config=config)
        print(f"number of parameters in model after compression is: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
    model.to(device)
    noise_scheduler = create_noise_scheduler(config)
    model.load_state_dict(torch.load(load_pretrain_model_path))
    print(f"Loaded model from {load_pretrain_model_path}")
    print(f"Using device: {device}")

    # model = TimestepConditionedWrapper(model, config)
    # print("Enabled timestep-conditioned rank scheduling")
    # print(f"  Schedule: {config.rank_schedule}")
    # print(f"  Min ratio: {config.rank_min_ratio}")
    # print(f"  Max timesteps: {config.num_training_steps}")


    print_model_settings(model)
    print_noise_scheduler_settings(noise_scheduler)

    dataloader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size=0.1)
    # Limit the number of batches processed per timestep to speed up evaluation
    max_batches_per_timestep = 4
    loss_dict = eval_timestamp_loss(model, noise_scheduler, dataloader, device, max_batches_per_timestep=max_batches_per_timestep)
    print(loss_dict)
    torch.save(loss_dict, f"Full_rank_timestep_loss_{load_pretrain_model_path.parent.name}_{load_pretrain_model_path.stem}.pt")
    plot_loss_dict(loss_dict, title="Loss vs Timestep", xlabel="Timestep", ylabel="Avg Loss", save_path=f"Full_rank_timestep_loss_{load_pretrain_model_path.parent.name}_{load_pretrain_model_path.stem}.pdf")


            