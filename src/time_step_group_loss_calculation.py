import os
from pathlib import Path
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings
from low_rank_compression import low_rank_layer_replacement
import torch
from tqdm import tqdm
import torch.nn.functional as F
from preprocessing import create_dataloader
from config import TrainingConfig
import matplotlib.pyplot as plt
import numpy as np
 

@torch.no_grad()
def calculate_time_step_group_loss(time_step_low_bound, time_step_high_bound, model, data_loader, noise_scheduler, device):
    model.eval()
    model.to(device)

    time_step_group_loss = 0
    time_step_group_loss_count = 0

    for batch in tqdm(data_loader, desc=f"Calculating time step group loss for time step group {time_step_low_bound} to {time_step_high_bound}"):
        clean_images = batch["img"].to(device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        batch_size = clean_images.shape[0]
        timesteps = torch.randint(time_step_low_bound, time_step_high_bound, (batch_size,), device=clean_images.device).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        if "label" in batch:
            class_labels = batch["label"].to(clean_images.device)
        else:
            class_labels = torch.zeros(clean_images.shape[0], dtype=torch.long, device=clean_images.device)


        pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]
        target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
        loss = F.mse_loss(pred, noise, reduction="none")
        loss = loss.mean()

        time_step_group_loss += loss.item()
        time_step_group_loss_count += 1

    return time_step_group_loss / time_step_group_loss_count



def main():
    config = TrainingConfig()
    config.train_batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_timestep_groups = 10
    training_group_boundaries = [0, 44, 123, 234, 371, 520, 667, 796, 897, 966, 1000]
    # training_group_boundaries= [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # checkpoint_list = ["0099", "0299", "0499", "0699", "0899", "1049", "1249", "1449", "1649", "1849", "2049"]
    checkpoint_list = ["0099", "0199", "0299", "0399", "0499", "0599", "0699", "0799", "0899", "0999", "1099", "1199"]
    load_checkpoint_folder = "DiT20250806_203705"

    # Initialize data storage for plotting
    # Dictionary to store losses: {timestep_group_idx: [loss_values_per_checkpoint]}
    timestep_group_losses = {i: [] for i in range(num_timestep_groups)}
    timestep_group_validation_losses = {i: [] for i in range(num_timestep_groups)}
    checkpoint_numbers = [int(ckpt) for ckpt in checkpoint_list]

    for checkpoint in checkpoint_list:
        load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / load_checkpoint_folder / f"model_{checkpoint}.pt"  
        
        
        train_loader = create_dataloader("uoft-cs/cifar10", "train", config)
        validation_loader = create_dataloader("uoft-cs/cifar10", "test", config)

        model = create_model(config)
        model.to(device)
        noise_scheduler = create_noise_scheduler(config)
        model.load_state_dict(torch.load(load_pretrain_model_path))
        print(f"Loaded model from {load_pretrain_model_path}")

        for i in range(num_timestep_groups):
            time_step_low_bound = training_group_boundaries[i]
            time_step_high_bound = training_group_boundaries[i + 1]

            time_step_group_loss = calculate_time_step_group_loss(time_step_low_bound, time_step_high_bound, model, train_loader, noise_scheduler, device)
            validation_loss = calculate_time_step_group_loss(time_step_low_bound, time_step_high_bound, model, validation_loader, noise_scheduler, device)
            print(f"Checkpoint {checkpoint} - Time step group {i} ({time_step_low_bound}-{time_step_high_bound}): Train={time_step_group_loss:.4f}, Val={validation_loss:.4f}")
            
            # Store the losses for plotting
            timestep_group_losses[i].append(time_step_group_loss)
            timestep_group_validation_losses[i].append(validation_loss)

    # Create the visualization
    plt.figure(figsize=(14, 10))
    
    # Get colors for each timestep group
    colors = plt.cm.tab10(range(num_timestep_groups))
    
    # Plot each timestep group as separate lines for both training and validation
    # Fixed: Include ALL timestep groups (was num_timestep_groups - 1)
    for i in range(num_timestep_groups):
        time_step_low = training_group_boundaries[i]
        time_step_high = training_group_boundaries[i + 1]
        color = colors[i]
        
        # Plot training loss
        plt.plot(checkpoint_numbers, timestep_group_losses[i], 
                marker='o', linewidth=2, markersize=6,
                label=f'Group {i} Train (t={time_step_low}-{time_step_high})', 
                linestyle='-', color=color)
        
        # Plot validation loss
        plt.plot(checkpoint_numbers, timestep_group_validation_losses[i], 
                marker='s', linewidth=2, markersize=6,
                label=f'Group {i} Val (t={time_step_low}-{time_step_high})', 
                linestyle='--', color=color)
    
    plt.xlabel('Checkpoint', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Timestep Group Loss vs Checkpoint (Training vs Validation)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent.parent / "logs" / load_checkpoint_folder / "timestep_group_loss_plot_train_val_uniform.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also display the plot
    plt.show()

    # Calculate percentage changes between consecutive checkpoints
    timestep_group_train_pct_changes = {i: [] for i in range(num_timestep_groups)}
    timestep_group_val_pct_changes = {i: [] for i in range(num_timestep_groups)}
    
    for i in range(num_timestep_groups):
        train_losses = timestep_group_losses[i]
        val_losses = timestep_group_validation_losses[i]
        
        # Calculate percentage changes (skip first checkpoint as there's no previous one)
        for j in range(1, len(train_losses)):
            # Training percentage change
            train_pct_change = ((train_losses[j] - train_losses[j-1]) / train_losses[j-1]) * 100
            timestep_group_train_pct_changes[i].append(train_pct_change)
            
            # Validation percentage change
            val_pct_change = ((val_losses[j] - val_losses[j-1]) / val_losses[j-1]) * 100
            timestep_group_val_pct_changes[i].append(val_pct_change)
    
    # Create percentage change visualization
    plt.figure(figsize=(14, 10))
    
    # Checkpoint numbers for percentage change (excluding first checkpoint)
    pct_change_checkpoints = checkpoint_numbers[1:]
    
    # Plot percentage changes for each timestep group
    for i in range(num_timestep_groups):
        time_step_low = training_group_boundaries[i]
        time_step_high = training_group_boundaries[i + 1]
        color = colors[i]
        
        # Plot training percentage change
        plt.plot(pct_change_checkpoints, timestep_group_train_pct_changes[i], 
                marker='o', linewidth=2, markersize=6,
                label=f'Group {i} Train (t={time_step_low}-{time_step_high})', 
                linestyle='-', color=color)
        
        # Plot validation percentage change
        plt.plot(pct_change_checkpoints, timestep_group_val_pct_changes[i], 
                marker='s', linewidth=2, markersize=6,
                label=f'Group {i} Val (t={time_step_low}-{time_step_high})', 
                linestyle='--', color=color)
    
    # Add horizontal line at 0% for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.xlabel('Checkpoint', fontsize=12)
    plt.ylabel('Percentage Change (%)', fontsize=12)
    plt.title('Percentage Change in Timestep Group Loss Between Consecutive Checkpoints', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the percentage change plot
    pct_change_output_path = Path(__file__).parent.parent / "logs" / load_checkpoint_folder / "timestep_group_loss_percentage_change_plot.png"
    plt.savefig(pct_change_output_path, dpi=300, bbox_inches='tight')
    print(f"Percentage change plot saved to {pct_change_output_path}")
    
    # Also display the percentage change plot
    plt.show()

    # Print percentage change statistics for each timestep group
    print("\nPercentage Change Statistics:")
    print("=" * 80)
    for i in range(num_timestep_groups):
        time_step_low = training_group_boundaries[i]
        time_step_high = training_group_boundaries[i + 1]
        
        train_changes = timestep_group_train_pct_changes[i]
        val_changes = timestep_group_val_pct_changes[i]
        
        print(f"\nGroup {i} (t={time_step_low}-{time_step_high}):")
        print(f"  Training - Mean: {np.mean(train_changes):.2f}%, Std: {np.std(train_changes):.2f}%")
        print(f"  Validation - Mean: {np.mean(val_changes):.2f}%, Std: {np.std(val_changes):.2f}%")
        print(f"  Training changes: {[f'{x:.2f}%' for x in train_changes]}")
        print(f"  Validation changes: {[f'{x:.2f}%' for x in val_changes]}")


        
if __name__ == "__main__":
    main()