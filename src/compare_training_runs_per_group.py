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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Note: For PNG export, you may need to install kaleido: pip install kaleido




@torch.no_grad()
def calculate_time_step_group_loss(time_step_low_bound, time_step_high_bound, model, data_loader, noise_scheduler, device):
    """Calculate training loss for a specific timestep group"""
    model.eval()
    model.to(device)

    time_step_group_loss = 0
    time_step_group_loss_count = 0

    for batch in tqdm(data_loader, desc=f"Calculating loss for timestep group {time_step_low_bound}-{time_step_high_bound}", leave=False):
        clean_images = batch["img"].to(device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        batch_size = clean_images.shape[0]
        timesteps = torch.randint(time_step_low_bound, time_step_high_bound, (batch_size,), device=clean_images.device).long()


        # target = noise_scheduler.get_velocity(clean_images, noise, timesteps)

        target = noise

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        if "label" in batch:
            class_labels = batch["label"].to(clean_images.device)
        else:
            class_labels = torch.zeros(clean_images.shape[0], dtype=torch.long, device=clean_images.device)

        pred = model(noisy_images, timesteps, class_labels, return_dict=False)[0]
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.mean()

        time_step_group_loss += loss.item()
        time_step_group_loss_count += 1

    return time_step_group_loss / time_step_group_loss_count


def load_training_runs_config():
    """Configure the training runs to compare"""
    training_runs = {
        # "Baseline": {
        #     "folder": "DiT20250815_133251",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749"]
        #     # "checkpoints": ["0049", "0099", "0149", "0199"]
        # },

        # "epsilon reverse uni sample":{
        #     "folder": "DiT20250821_005658",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749"]
        # },

        # "FF epsilon reverse":{
        #     "folder": "DiT20250822_030828",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749"]
        # },

        # "FF epsilon":{
        #     "folder": "DiT20250822_025707",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749"]
        # },


        # "Currc Start high": {
        #     "folder": "DiT20250815_133041", 
        #     "checkpoints": ["0099", "0199", "0299", "0399", "0499", "0599", "0699", "0799", "0899", "0999", "1099", "1199", "1299", "1399", "1499", "1599", "1699", "1799", "1899", "1999", "2099"],
        # },

        # "Currc Start low": {
        #     "folder": "DiT20250815_132803",
        #     "checkpoints": ["0099", "0199", "0299", "0399", "0499", "0599", "0699", "0799", "0899", "0999", "1099", "1199", "1299", "1399", "1499", "1599", "1699", "1799", "1899", "1999", "2099"],
        # },

        # "floor 60%": {
        #     "folder": "DiT20250817_152007",
        #     "checkpoints": ["0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049"],
        # },

        # "floor 80%": {
        #     "folder": "DiT20250817_151731",
        #     "checkpoints": ["0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049"],
        # },


        # "floor 60% second round":{
        #     "folder": "DiT20250818_021738",
        #     "checkpoints": ["0299", "0349", "0399", "0449", "0499", "0549", "0599"],
        # },

        # "v-prediction full rank baseline":{
        #     "folder": "DiT20250820_181052",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699"]
        # }, 

        # "v-prediction galor baseline":{
        #     "folder": "DiT20250821_002417",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549"]
        # },

        # "V galore KD 50":{
        #     "folder": "DiT20250821_011658",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049", "1099", "1149", "1199"]
        # },

        # "V galore KD 10":{
        #     "folder": "DiT20250821_012447",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949"]
        # },

        # "v reverse uni sample":{
        #     "folder": "DiT20250821_005914",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549"]
        # },

        # "v reverse FF uni sample":{
        #     "folder": "DiT20250822_143237",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499"]
        # },

        # "v reverse FF uni sample":{
        #     "folder": "DiT20250822_192914",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134", "0149", "0164", "0179", "0194", "0209", "0224", "0239", "0254", "0269", "0284", "0299", "0314", "0329", "0344", "0359"]
        # },

        # "v reverse uni sample old":{
        #     "folder": "DiT20250822_192436",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134", "0149", "0164", "0179", "0194", "0209", "0224", "0239", "0254", "0269", "0284"]
        # },

        # "full rank curriculum epsilon prediction uni sampling":{
        #     "folder": "DiT20250821_161119",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249", "0299", "0349"]
        # },  

        # "full rank baseline epsilon prediction uni sampling":{
        #     "folder": "DiT20250821_202015",
        #     # "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049", "1099", "1149", "1199"]
        #     "checkpoints": ["0049", "0099", "0149", "0199"]
        # },  

        # "full rank FF unifrom sampling 0.01":{
        #     "folder": "DiT20250822_230533",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134", "0149", "0164", "0179", "0194", "0209", "0224", "0239", "0254", "0269", "0284", "0299", "0314"]
        # },

        # "full rank FF unifrom sampling 0.05":{
        #     "folder": "DiT20250823_005407",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134", "0149", "0164"]
        # },

        # "full rank FF unifrom sampling 0.05 five groups":{
        #     "folder": "DiT20250823_105806",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134", "0149", "0164", "0179", "0194", "0209",]
        # },

        # "full rank fix round 5 group":{
        #     "folder": "DiT20250823_141048",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134", "0149"]
        # },

        # "full rank fix round 5 group start 44":{
        #     "folder": "DiT20250823_144235",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },


        # "full rank fix round 5 group start 29":{
        #     "folder": "DiT20250823_143032",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", ]
        # },

        # "full rank fix round 5 group start 44 start high":{
        #     "folder": "DiT20250823_151919",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104"]
        # },

        # "full rank fix gradual 150":{
        #     "folder": "DiT20250823_195526",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },

        # "full rank fix gradual 45":{
        #     "folder": "DiT20250823_195610",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },

        # "full rank fix gradual 45 faster":{
        #     "folder": "DiT20250823_211341",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },

        # "full rank fix gradual 150 faster":{
        #     "folder": "DiT20250823_211856",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104"]
        # },

        # "full rank fix gradual 45 preview 0.3":{
        #     "folder": "DiT20250823_220826",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },

        # "full rank fix gradual 45 preview 0.5":{
        #     "folder": "DiT20250823_220957",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },

        # "full dataset baseline":{
        #     "folder": "DiT20250823_154440",
        #     "checkpoints": ["0014", "0029", "0044", "0059"]
        # },

        # "full dataset start low fix 15":{
        #     "folder": "DiT20250823_154059",
        #     "checkpoints": ["0014", "0029", "0044", "0059"]
        # },

        # "full rank fix gradual reverse 966 preview 0.5":{
        #     "folder": "DiT20250824_000320",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },

        # "full rank fix gradual reverse 844 preview 0.5":{
        #     "folder": "DiT20250824_000436",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119"]
        # },


        # "full rank fix gradual 100 preview 0.3":{
        #     "folder": "DiT20250824_010517",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049", "1099", "1149", "1199"]
        # },

        # "full rank fix gradual 200 preview 0.3":{
        #     "folder": "DiT20250824_055017",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049", "1099", "1149", "1199"]
        # },

        # "full rank fix gradual 100 preview 0.15":{
        #     "folder": "DiT20250824_010653",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049", "1099", "1149", "1199"]
        # },

        # "full rank fix gradual 200 preview 0.15":{
        #     "folder": "DiT20250824_055446",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999", "1049", "1099", "1149", "1199"]
        # },

        # "full rank fix gradual 400 preview 0.3":{
        #     "folder": "DiT20250824_100815",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999"]
        # },

        # "full rank fix gradual 400 preview 0.15":{
        #     "folder": "DiT20250824_101649",
        #     "checkpoints": ["0049", "0099", "0149","0199","0249" , "0299", "0349","0399", "0449", "0499", "0549",  "0599", "0649", "0699", "0749", "0799", "0849", "0899", "0949", "0999"]
        # },

        # "full rank fix gradual 100 preview 0.3":{
        #     "folder": "DiT20250824_134940",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]        
        # },

        # "full rank fix gradual 100 preview 0.15":{
        #     "folder": "DiT20250824_135324",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]        
        # },

        # "full rank fix start low gradual 100 preview 0.1":{
        #     "folder": "DiT20250824_151135",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]        
        # },

        # "full rank fix start low gradual 100 preview 0.15":{
        #     "folder": "DiT20250824_151300",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]        
        # },

        # "full rank fix start low 44 gradual 120 preview 0 - 0.1":{
        #     "folder": "DiT20250824_163306",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]        
        # },

        # "full rank fix start low 44 gradual 120 preview 0.05 - 0.1":{
        #     "folder": "DiT20250824_163307",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]        
        # },

        # "Galore fix start low 44 gradual 120 preview 0 - 0.1":{
        #     "folder": "DiT20250824_171633",
        #     "checkpoints": ["0014", "0044", "0074", "0104", "0134", "0164", "0194", "0224", "0254", "0284", "0314", "0344", "0374", "0404", "0434", "0464", "0494", "0524", "0554", "0584", "0614", "0644"]
        # },


        # "Galore fix start low 44 gradual 120 preview 0.05 - 0.1":{
        #     "folder": "DiT20250824_171643",
        #     "checkpoints": ["0014", "0044", "0074", "0104", "0134", "0164", "0194", "0224", "0254", "0284", "0314", "0344", "0374", "0404", "0434", "0464", "0494", "0524", "0554", "0584", "0614", "0644"]
        # },

        # "Batch 128 Galore fix start low 44 gradual 120 preview 0 - 0.1":{
        #     "folder": "DiT20250824_210421",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]
        # },

        # "Batch 128 Galore baseline":{
        #     "folder": "DiT20250824_194516",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]
        # },

        # "Batch 128 Galore full dataset fix start low 44 gradual 120 preview 0.05 - 0.1":{
        #     "folder": "DiT20250824_221139",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]
        # },

        # "Batch 128 Galore full dataset baseline":{
        #     "folder": "DiT20250824_221301",
        #     "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]
        # },

        "low rank parameterisation baseline":{
            "folder": "DiT20250825_002435",
            "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]
        },

        "low rank parameterisation fix start low 44 gradual 120 preview 0.05 - 0.1":{
            "folder": "DiT20250825_004507",
            "checkpoints": ["0014", "0029", "0044", "0059", "0074", "0089", "0104", "0119", "0134"]
        },


        # Add more training runs as needed
        # "Low Rank R=32": {
        #     "folder": "another_folder",
        #     "checkpoints": ["0099", "0199", "0299", "0399", "0499"],
        #     "low_rank": True,
        #     "rank": 32
        # }
    }
    return training_runs


def main():
    config = TrainingConfig()
    config.train_batch_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestep group boundaries
    num_timestep_groups = 10
    training_group_boundaries = [0, 44, 123, 234, 371, 520, 667, 796, 897, 966, 1000]

    # num_timestep_groups = 5
    # training_group_boundaries =  [0, 133, 372, 653, 881, 1000]
    
    # Load training runs configuration
    training_runs = load_training_runs_config()
    
    # Create data loader
    train_loader = create_dataloader("uoft-cs/cifar10", "train", config, subset_size= 0.3)
    noise_scheduler = create_noise_scheduler(config)
    
    # Storage for all results: {run_name: {timestep_group_idx: [losses_per_checkpoint]}}
    all_results = {}
    
    print("Processing training runs...")
    print("=" * 80)        

    tansform_dict = {}

    # Process each training run
    for run_name, run_config in training_runs.items():
        print(f"\nProcessing {run_name}...")


        curriculum_transform_path = Path(__file__).parent.parent / "logs" / run_config["folder"] / "curriculum_swap_epoch_list.pt"
        if curriculum_transform_path.exists():
            curriculum_transform = torch.load(curriculum_transform_path)
            print(f"Curriculum transform for {run_name}: {curriculum_transform}")
            if "reverse" not in run_name.lower():
                curriculum_transform.reverse()
            tansform_dict[run_name] = curriculum_transform

            print(tansform_dict)


        results_path = Path(__file__).parent.parent / "logs" / run_config["folder"] / f"all_results_{run_name}.pt"
        if results_path.exists():
            all_results[run_name] = torch.load(results_path)
            print(f"Loaded existing results for {run_name}")
            continue


        # Initialize storage for thrun_nameis run
        all_results[run_name] = {i: [] for i in range(num_timestep_groups)}
        checkpoint_numbers = [int(ckpt) for ckpt in run_config["checkpoints"]]
        
        # Process each checkpoint in this run
        for checkpoint in run_config["checkpoints"]:
            load_pretrain_model_path = Path(__file__).parent.parent / "logs" / run_config["folder"] / f"model_{checkpoint}.pt"
            
            if not load_pretrain_model_path.exists():
                print(f"Warning: Checkpoint {checkpoint} not found for {run_name} at {load_pretrain_model_path}")
                continue
                
            # Create and load model
            model = create_model(config)
            
            # Apply low rank compression if specified
            model = low_rank_layer_replacement(model, percentage=0.25)
            print(f"number of parameters in model after compression is: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

                
            model.to(device)
            model.load_state_dict(torch.load(load_pretrain_model_path))
            print(f"  Loaded checkpoint {checkpoint} from {run_config['folder']}")
            
            # Calculate loss for each timestep group
            for group_idx in range(num_timestep_groups):
                time_step_low_bound = training_group_boundaries[group_idx]
                time_step_high_bound = training_group_boundaries[group_idx + 1]
                
                group_loss = calculate_time_step_group_loss(
                    time_step_low_bound, time_step_high_bound, 
                    model, train_loader, noise_scheduler, device
                )
                
                all_results[run_name][group_idx].append(group_loss)

            print(f"    Completed checkpoint {checkpoint}")
            torch.save(all_results[run_name], Path(__file__).parent.parent / "logs" / run_config["folder"] / f"all_results_{run_name}.pt")
    
    print("\nGenerating plots...")
    print("=" * 80)
    
    # Define colors and line styles for plotly
    plotly_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['solid', 'dash', 'dashdot', 'dot']
    plotly_markers = ['circle', 'square', 'triangle-up', 'triangle-down', 'diamond', 'cross', 'x', 'star']
    
    for group_idx in range(num_timestep_groups):
        time_step_low = training_group_boundaries[group_idx]
        time_step_high = training_group_boundaries[group_idx + 1]

        filter_threshold = 1
        
        # Set filtering threshold based on group
        # if group_idx == 0:
        #     filter_threshold = 0.5
        # elif group_idx == 1:
        #     filter_threshold = 0.3
        # elif group_idx == 2:
        #     filter_threshold = 0.2
        # else:
        #     filter_threshold = 0.1

        # if group_idx == 0:
        #     filter_threshold = 0.8
        # elif group_idx == 1 or group_idx == 9:
        #     filter_threshold = 0.5
        # elif group_idx == 2 or group_idx == 8:
        #     filter_threshold = 0.3
        # else:
        #     filter_threshold = 0.1
        
        # Create plotly figure
        fig = go.Figure()
        
        # Plot each training run for this timestep group
        for run_idx, (run_name, run_config) in enumerate(training_runs.items()):
            if group_idx in all_results[run_name] and all_results[run_name][group_idx]:
                checkpoint_numbers = [int(ckpt) for ckpt in run_config["checkpoints"]]
                losses = all_results[run_name][group_idx]
                
                # Ensure we have the same number of checkpoints and losses
                min_length = min(len(checkpoint_numbers), len(losses))
                checkpoint_numbers = checkpoint_numbers[:min_length]
                if run_name == "full rank fix round 5 group start 44":
                    checkpoint_numbers = [ckpt + 45 for ckpt in checkpoint_numbers]
                if run_name == "full rank fix round 5 group start 29":
                    checkpoint_numbers = [ckpt + 30 for ckpt in checkpoint_numbers]
                if run_name == "full rank fix round 5 group start 44 start high":
                    checkpoint_numbers = [ckpt + 45 for ckpt in checkpoint_numbers]
                losses = losses[:min_length]
                
                # Filter out values above threshold
                filtered_checkpoints = []
                filtered_losses = []
                filtered_count = 0
                for ckpt, loss in zip(checkpoint_numbers, losses):
                    if loss <= filter_threshold:
                        filtered_checkpoints.append(ckpt)
                        filtered_losses.append(loss)
                    else:
                        filtered_count += 1
                
                # Log filtering information
                if filtered_count > 0:
                    print(f"  Group {group_idx} - {run_name}: Filtered out {filtered_count}/{len(losses)} points above {filter_threshold}")
                
                # Only plot if we have data after filtering
                if filtered_checkpoints:
                    color = plotly_colors[run_idx % len(plotly_colors)]
                    line_style = line_styles[run_idx % len(line_styles)]
                    marker = plotly_markers[run_idx % len(plotly_markers)]
                    
                    fig.add_trace(go.Scatter(
                        x=filtered_checkpoints,
                        y=filtered_losses,
                        mode='lines+markers',
                        name=run_name,
                        line=dict(color=color, width=2.5, dash=line_style),
                        marker=dict(symbol=marker, size=8, color=color)
                    ))
        
        # Add vertical lines for curriculum swaps
        vertical_line_colors = ['red', 'blue', 'green', 'purple']
        vertical_line_styles = ['dash', 'dashdot', 'dot', 'solid']
        
        for run_idx, run_name in enumerate(tansform_dict.keys()):
            print(run_name)
            if group_idx in tansform_dict[run_name]:
                transform_location_for_group = tansform_dict[run_name][group_idx]
            else:
                continue
            v_line_style = vertical_line_styles[run_idx % len(vertical_line_styles)]
            v_color = vertical_line_colors[run_idx % len(vertical_line_colors)]
            
            fig.add_vline(
                x=transform_location_for_group,
                line=dict(color=v_color, width=3, dash=v_line_style),
                opacity=0.7,
                annotation_text=f'Curriculum Swap {run_name}',
                annotation_position="top"
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Training Loss Comparison - Timestep Group {group_idx}<br>(Timesteps {time_step_low}-{time_step_high}) [Filtered: loss ≤ {filter_threshold}]',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title=dict(text='Checkpoint', font=dict(size=14)),
            yaxis_title=dict(text='Training Loss (log scale)', font=dict(size=14)),
            yaxis_type="log",
            width=1200,
            height=800,
            showlegend=True,
            legend=dict(font=dict(size=12)),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )

        # Save both PNG and HTML versions
        output_dir = Path(__file__).parent.parent / "logs" / "DiT20250825_004507"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        png_path = output_dir / f"timestep_group_{group_idx}_comparison.png"
        html_path = output_dir / f"timestep_group_{group_idx}_comparison.html"
        
        # Save PNG (with error handling for kaleido dependency)
        try:
            fig.write_image(png_path, width=1200, height=800, scale=2)
        except Exception as e:
            print(f"Warning: Could not save PNG file. Install kaleido with 'pip install kaleido'. Error: {e}")
        
        # Save HTML
        fig.write_html(html_path)
        
        print(f"Saved plot for timestep group {group_idx} to {png_path} and {html_path}")
        
        # Show plot
        # fig.show()
    
    # Create a summary plot with all timestep groups for the first training run (for reference)
    print("\nGenerating summary plot...")
    first_run_name = list(training_runs.keys())[0]
    first_run_config = training_runs[first_run_name]
    checkpoint_numbers = [int(ckpt) for ckpt in first_run_config["checkpoints"]]
    
    # Create plotly summary figure
    summary_fig = go.Figure()
    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for group_idx in range(num_timestep_groups):
        time_step_low = training_group_boundaries[group_idx]
        time_step_high = training_group_boundaries[group_idx + 1]
        
        if group_idx in all_results[first_run_name] and all_results[first_run_name][group_idx]:
            losses = all_results[first_run_name][group_idx]
            min_length = min(len(checkpoint_numbers), len(losses))
            
            summary_fig.add_trace(go.Scatter(
                x=checkpoint_numbers[:min_length],
                y=losses[:min_length],
                mode='lines+markers',
                name=f'Group {group_idx} (t={time_step_low}-{time_step_high})',
                line=dict(color=group_colors[group_idx % len(group_colors)], width=2),
                marker=dict(symbol='circle', size=6, color=group_colors[group_idx % len(group_colors)])
            ))
    
    # Update summary layout
    summary_fig.update_layout(
        title=dict(
            text=f'All Timestep Groups - {first_run_name}',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title=dict(text='Checkpoint', font=dict(size=14)),
        yaxis_title=dict(text='Training Loss (log scale)', font=dict(size=14)),
        yaxis_type="log",
        width=1400,
        height=1000,
        showlegend=True,
        legend=dict(font=dict(size=12)),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    # Save summary plot (both PNG and HTML)
    output_dir = Path(__file__).parent.parent / "logs" / "DiT20250825_004507"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_png_path = output_dir / f"all_timestep_groups_{first_run_name.replace(' ', '_')}.png"
    summary_html_path = output_dir / f"all_timestep_groups_{first_run_name.replace(' ', '_')}.html"
    
    # Save PNG and HTML (with error handling for kaleido dependency)
    try:
        summary_fig.write_image(summary_png_path, width=1400, height=1000, scale=2)
    except Exception as e:
        print(f"Warning: Could not save summary PNG file. Install kaleido with 'pip install kaleido'. Error: {e}")
    
    summary_fig.write_html(summary_html_path)
    
    print(f"Saved summary plot to {summary_png_path} and {summary_html_path}")
    # summary_fig.show()
    
    # Print statistics
    print("\nTraining Loss Statistics:")
    print("=" * 80)
    for run_name, run_results in all_results.items():
        print(f"\n{run_name}:")
        for group_idx in range(num_timestep_groups):
            if group_idx in run_results and run_results[group_idx]:
                losses = run_results[group_idx]
                time_step_low = training_group_boundaries[group_idx]
                time_step_high = training_group_boundaries[group_idx + 1]
                
                print(f"  Group {group_idx} (t={time_step_low}-{time_step_high}): "
                      f"Mean={np.mean(losses):.4f}, Std={np.std(losses):.4f}, "
                      f"Final={losses[-1]:.4f}")


if __name__ == "__main__":
    main() 