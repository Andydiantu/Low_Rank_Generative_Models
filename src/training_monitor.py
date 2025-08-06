import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import deque

class TrainingMonitor:
    def __init__(self, patience, num_timestep_groups, k=10, start_from_low=True):
        self.patience = patience
        self.k = k  # Number of steps to track for running mean
        self.recent_losses = deque(maxlen=k)  # Circular buffer for past k losses
        self.best_running_mean = float('inf')
        self.running_mean = float('inf')
        self.counter = 0
        self.num_timestep_groups = num_timestep_groups
        self.start_from_low = start_from_low
        
        # Set initial timestep group based on direction
        if start_from_low:
            # Start from 0, progress to 966
            self.current_timestep_groups = 0
        else:
            # Start from 966, progress to 0 (original behavior)
            self.current_timestep_groups = num_timestep_groups - 1
            
        self.training_group_boundaries = [0, 44, 123, 234,371, 520, 667, 796, 897, 966]

    def __call__(self, loss):
        # Add current loss to the buffer
        self.recent_losses.append(loss)
        
        # Calculate running mean if we have enough samples
        if len(self.recent_losses) >= self.k:
            self.running_mean = sum(self.recent_losses) / len(self.recent_losses)
            
            # Compare current loss to running mean
            if self.running_mean < self.best_running_mean:
                self.best_running_mean = self.running_mean
                self.counter = 0
            else:
                self.counter += 1
                print(f"counter: {self.counter}")
                if self.counter >= self.patience:
                    # Reset the counter and clear the buffer
                    self.counter = 0
                    self.recent_losses.clear()
                    self.running_mean = float('inf')
                    
                    # Progress based on direction
                    if self.start_from_low:
                        # Progress to higher timestep groups (0 → 966)
                        self.current_timestep_groups += 1
                    else:
                        # Progress to lower timestep groups (966 → 0)
                        self.current_timestep_groups -= 1
                    return True
        else:
            # Not enough samples yet, just reset counter
            self.counter = 0
            
        return False

    def get_current_timestep_groups_low_bound(self):
        if self.start_from_low:
            # Forward progression: return current boundary
            return self.training_group_boundaries[self.current_timestep_groups]
        else:
            # Backward progression: return previous boundary
            return self.training_group_boundaries[self.current_timestep_groups - 1]

    def get_current_timestep_groups_high_bound(self):
        if self.start_from_low:
            # Forward progression
            if self.current_timestep_groups == self.num_timestep_groups - 1:
                return 1000
            else:
                return self.training_group_boundaries[self.current_timestep_groups + 1]
        else:
            # Backward progression (original behavior)
            if self.current_timestep_groups == self.num_timestep_groups:
                return 1000
            else:
                return self.training_group_boundaries[self.current_timestep_groups]
            
    def get_trained_timesteps_boundaries(self):
        if self.start_from_low:
            return [0, self.get_current_timestep_groups_low_bound()]
        else:
            return [self.get_current_timestep_groups_high_bound(), 1000]


    def get_if_curriculum_learning_is_done(self):
        if self.start_from_low:
            return self.current_timestep_groups > self.num_timestep_groups - 1
        else:
            return self.current_timestep_groups == 0