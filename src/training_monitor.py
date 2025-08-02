import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import deque

class TrainingMonitor:
    def __init__(self, patience, num_timestep_groups, k=10):
        self.patience = patience
        self.k = k  # Number of steps to track for running mean
        self.recent_losses = deque(maxlen=k)  # Circular buffer for past k losses
        self.best_running_mean = float('inf')
        self.running_mean = float('inf')
        self.counter = 0
        self.num_timestep_groups = num_timestep_groups
        self.current_timestep_groups = num_timestep_groups -1 
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
                    self.current_timestep_groups -= 1
                    return True
        else:
            # Not enough samples yet, just reset counter
            self.counter = 0
            
        return False

    def get_current_timestep_groups_low_bound(self):
        return self.training_group_boundaries[self.current_timestep_groups-1]

    def get_current_timestep_groups_high_bound(self):
        if self.current_timestep_groups == self.num_timestep_groups:
            return 1000
        else:
            return self.training_group_boundaries[self.current_timestep_groups]