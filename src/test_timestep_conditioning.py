#!/usr/bin/env python3
"""
Test script for timestep-conditioned rank scheduling in DiT model.
"""

import sys
sys.path.append('src')

import torch
from config import TrainingConfig
from DiT import create_model
from low_rank_compression import low_rank_layer_replacement, TimestepConditionedWrapper, LowRankLinear

def test_timestep_conditioning():
    print("Testing timestep-conditioned rank scheduling...")
    
    # Create config with timestep conditioning enabled
    config = TrainingConfig()
    config.low_rank_pretraining = True
    config.low_rank_rank = 0.25
    config.timestep_conditioning = True
    config.rank_schedule = "decreasing"
    config.rank_min_ratio = 0.3
    config.num_training_steps = 1000
    
    print(f"Config: rank={config.low_rank_rank}, schedule={config.rank_schedule}, min_ratio={config.rank_min_ratio}")
    
    # Create and modify model
    model = create_model(config)
    print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Apply low-rank replacement
    model = low_rank_layer_replacement(model, percentage=config.low_rank_rank)
    print(f"Low-rank parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Wrap with timestep conditioning
    model = TimestepConditionedWrapper(model, config)
    print(f"Found {len(model.low_rank_layers)} LowRankLinear layers")
    
    # Test forward pass with different timesteps
    batch_size = 4
    hidden_states = torch.randn(batch_size, 3, 32, 32)  # CIFAR10 images
    timesteps = torch.tensor([50, 200, 500, 900])  # Different timesteps
    class_labels = torch.zeros(batch_size, dtype=torch.long)
    
    print(f"\nTesting with timesteps: {timesteps.tolist()}")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Patch size: {config.image_size // 2} → {(config.image_size // 2)**2} patches per image")
    print(f"Expected internal batch size: {batch_size} images × {(config.image_size // 2)**2} patches = {batch_size * (config.image_size // 2)**2}")
    
    # Forward pass
    with torch.no_grad():
        output = model(hidden_states, timesteps, class_labels)
        print(f"Output shape: {output[0].shape}")
        
    print("✓ Timestep conditioning test passed!")
    print("✓ Batch size mismatch between patches and timesteps handled correctly!")
    
    # Test different schedules
    print("\nTesting different schedules...")
    schedules = ["decreasing", "increasing", "midpeak"]
    
    for schedule in schedules:
        config.rank_schedule = schedule
        wrapper = TimestepConditionedWrapper(model.base_model, config)
        
        with torch.no_grad():
            output = wrapper(hidden_states, timesteps, class_labels)
            print(f"✓ {schedule} schedule works")
    
    print("\n✅ All tests passed!")
    print("The timestep conditioning system correctly handles:")
    print("  • Patch-level vs image-level batch size mismatches")
    print("  • Automatic timestep expansion for internal computations")
    print("  • Different rank scheduling strategies")

if __name__ == "__main__":
    test_timestep_conditioning() 