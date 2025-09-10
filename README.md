# Low Rank Generative Models

This repository contains the codebase for my MSc Dissertation at Imperial College London, focusing on exploring the application of low-rank methods to generative models, specifically diffusion models.

## Overview

We propose a **timestep adaptive low-rank diffusion model** that activates different amounts of rank for different timesteps during the diffusion process. This approach leverages the empirical observation that both weight matrices and gradients exhibit lower intrinsic rank as timesteps increase.

## Key Contributions

### Empirical Observation
Our research is motivated by the observation that as timesteps increase in diffusion models, both weight matrices and gradients demonstrate lower intrinsic rank, suggesting an opportunity for adaptive rank optimization.

### Timestep Adaptive Low Rank Diffusion Model
Our method employs:
- **Training time masking**: Training different timesteps with different activated ranks
- **Inference optimization**: Utilizing slicing techniques to reduce computational overhead while maintaining model performance

## Quick Start

### Running Experiments
To recreate our experiments for each dataset, simply run the provided shell scripts:

```bash
# For CelebA dataset
./Celeb_A.sh

# For CIFAR-10 dataset  
./CIFAR_10.sh

# For ImageNet-128 dataset
./imagenet_128.sh
```

Each script will run three baseline comparisons:
- Full rank baseline
- 50% static low rank baseline  
- 50% adaptive low rank baseline

### Custom Configuration
Alternatively, you can run experiments with custom configurations:

```bash
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)

python -u src/DiT_trainer.py \
    --set model=DiT-B/2 \
    --set dataset=imagenet \
    --set low_rank_pretraining=True \
    --set timestep_conditioning=True \
    --set low_rank_rank=0.5 \
    --set timestep_conditioning_match_type=activated \
    --set curriculum_learning=True \
    --set curriculum_learning_start_from_low=False \
    2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `model` | The model architecture to use (e.g., DiT-B/2) |
| `dataset` | Target dataset (imagenet, cifar10, celeba) |
| `low_rank_pretraining` | Enable low-rank pretraining |
| `timestep_conditioning` | Enable adaptive timestep low-rank parameterization |
| `low_rank_rank` | Rank ratio to use (0.0-1.0) |
| `timestep_conditioning_match_type` | Conditioning match type:<br/>• `activated`: ISO inference compute<br/>• `total`: Match total number of parameters |
| `curriculum_learning` | Enable curriculum learning |
| `curriculum_learning_start_from_low` | Start curriculum from low timesteps |

## Repository Structure

```
├── src/
│   ├── DiT_trainer.py          # Main training script
│   ├── DiT.py                  # DiT model implementation
│   ├── config.py               # Configuration management
│   └── generate_from_checkpoint.py  # Inference script
├── Celeb_A.sh                  # CelebA experiment script
├── CIFAR_10.sh                 # CIFAR-10 experiment script
├── imagenet_128.sh             # ImageNet-128 experiment script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

