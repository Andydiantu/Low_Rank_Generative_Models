from itertools import cycle
from low_rank_compression import label_low_rank_gradient_layers
from galore_torch import GaLoreEvalAdamW
import torch
import torch.nn.functional as F
from config import TrainingConfig
from preprocessing import create_dataloader
from DiT import create_model, create_noise_scheduler, print_model_settings, print_noise_scheduler_settings
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def eval_galore_projection_loss(max_timestep, min_timestep, model, dataloader, noise_scheduler, galoer_round,device):
    decompose_step_loss = []
    subspace_drift_loss = []

    for galore_round in range(galoer_round):
        galore_params, regular_params = label_low_rank_gradient_layers(model)
        param_to_name = {param: name for name, param in model.named_parameters()}

        regular_param_names = [param_to_name[p] for p in regular_params]
        galore_param_names = [param_to_name[p] for p in galore_params]

        param_groups = [
        {"params": regular_params, "param_names": regular_param_names},
        {
            "params": galore_params,
            "rank": 128,
            "update_proj_gap": 200,
            "scale": 1.0,
            "proj_type": "std",
            "param_names": galore_param_names,
        },]

        optimizer = GaLoreEvalAdamW(param_groups, lr=0.0001, weight_decay=0.0)

        projection_step_loss = []

        for i, batch in enumerate(tqdm(cycle(dataloader), desc="Evaluating Galore Projection Loss")):
            

            if i >= 200:
                break

            clean_images = batch["img"].to(device)
            batch_size = clean_images.shape[0]

            class_labels = None
            if "label" in batch:
                class_labels = batch["label"].to(device)
            else:
                raise ValueError("Label is not in the batch")

                
            timesteps = torch.randint(min_timestep, max_timestep, (batch_size,), device=clean_images.device).long()
            noise = torch.randn_like(clean_images, device=device)

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise, reduction="none")
            loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            _, projection_loss_dict = optimizer.step()
            optimizer.zero_grad()
            total_projection_loss = 0.0

            num_layer_count = 0
            for param_name, (err_F, err_cos) in projection_loss_dict.items():
                total_projection_loss += err_F
                num_layer_count += 1

            avg_projection_loss = total_projection_loss / num_layer_count

            if i == 0: 
                decompose_step_loss.append(avg_projection_loss.detach().cpu().numpy())
            else:
                projection_step_loss.append(avg_projection_loss.detach().cpu().numpy())
            
        projection_step_loss_mean = sum(projection_step_loss) / len(projection_step_loss)
        subspace_drift_loss.append(projection_step_loss_mean)


    return subspace_drift_loss, decompose_step_loss



def main():

    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_subspace_drift_loss = []
    total_decompose_step_loss = []

    checkpoint_list = ["0099", "0199", "0299", "0399", "0499", "0599", "0699", "0799", "0899", "0999", "1099", "1199", "1299", "1399", "1499", "1599", "1699", "1799", "1899", "1999", "2099", "2199"]

    for checkpoint in checkpoint_list:
        load_pretrain_model_path = Path(__file__).parent.parent / "logs"  / "DiT20250726_235632" / f"model_{checkpoint}.pt"  
        
        
        dataloader = create_dataloader("uoft-cs/cifar10", "train", config)
        model = create_model(config)
        model.to(device)
        noise_scheduler = create_noise_scheduler(config)
        model.load_state_dict(torch.load(load_pretrain_model_path))
        print(f"Loaded model from {load_pretrain_model_path}")

        subspace_drift_loss, decompose_step_loss = eval_galore_projection_loss(1000, 0, model, dataloader, noise_scheduler, 3, device)

        average_subspace_drift_loss = sum(subspace_drift_loss) / len(subspace_drift_loss)
        average_decompose_step_loss = sum(decompose_step_loss) / len(decompose_step_loss)

        total_subspace_drift_loss.append(average_subspace_drift_loss)
        total_decompose_step_loss.append(average_decompose_step_loss)

        print(f"Subspace drift loss: {average_subspace_drift_loss}")
        print(f"Decompose step loss: {average_decompose_step_loss}")
        
    checkpoint_nums = [int(checkpoint) for checkpoint in checkpoint_list]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Subspace Drift Loss', color=color1)
    line1 = ax1.plot(checkpoint_nums, total_subspace_drift_loss, color=color1, marker='o', linewidth=2, label='Subspace Drift Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Decompose Step Loss', color=color2)
    line2 = ax2.plot(checkpoint_nums, total_decompose_step_loss, color=color2, marker='s', linewidth=2, label='Decompose Step Loss')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('GaLore Projection Loss Evaluation vs Checkpoint', fontsize=14, pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent.parent / "galore_projection_loss_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()
        
if __name__ == "__main__":
    main()
        


            

                
                



