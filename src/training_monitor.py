import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import deque

class TrainingMonitor:
    def __init__(self, patience, num_timestep_groups, k=10, start_from_low=False, start_from_middle=False, middle_group_index=3, ema_alpha=0.1, ema_warmup=3):
        self.patience = patience
        self.k = k  # Number of steps to track for running mean
        self.recent_losses = deque(maxlen=k)  # Circular buffer for past k losses   
        self.best_running_mean = float('inf')
        self.running_mean = float('inf')
        self.counter = 0
        self.num_timestep_groups = num_timestep_groups
        self.start_from_low = start_from_low
        self.start_from_middle = start_from_middle

        self.ema_alpha = ema_alpha
        self.ema_warmup = ema_warmup
        self.ema_moving_average = None
        self.ema_counter = 0
        self.last_boundary = None
        self.EMA_list = []

        self.training_group_boundaries = [0, 44 , 123, 234, 371, 520, 667, 796, 897,967 , 1000]
        # self.training_group_boundaries = [0, 17, 44, 81 , 128, 185, 250, 323, 400, 481, 562, 641, 716, 783, 844, 895, 936, 967, 988, 999, 1000]
        # self.training_group_boundaries = [0, 133, 372, 653, 881, 1000]
        # self.training_group_boundaries = [0, 23, 64, 121, 194, 279, 374, 475, 578, 677, 767, 844, 907, 954, 985, 1000]
        # Initialize training state based on mode
        if start_from_middle:
            if middle_group_index is None:
                # Default to middle of available groups
                self.middle_group_index = num_timestep_groups // 2
            else:
                self.middle_group_index = middle_group_index
            
            self.current_timestep_groups = self.middle_group_index
            self.trained_groups = {self.middle_group_index}  # Track which groups have been trained
            self.next_direction = 'left'  # Start by going left first
            
        elif start_from_low:
            # Start from 0, progress to 966
            self.current_timestep_groups = 0
            self.trained_groups = {0}
        else:
            # Start from 966, progress to 0 (original behavior)
            self.current_timestep_groups = num_timestep_groups - 1
            self.trained_groups = {num_timestep_groups - 1}

    def reset_curriculum_learning(self):
        self.current_timestep_groups = self.num_timestep_groups - 1
        self.trained_groups = {self.num_timestep_groups - 1}
        self.next_direction = 'left'
        self.last_boundary = None
        self.ema_moving_average = None
        self.ema_counter = 0

    def __call__(self, loss):
        """Update training monitor with new loss and check if we should progress to next group."""
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
                    return self._progress_to_next_group()
        else:
            # Not enough samples yet, just reset counter
            self.counter = 0
            
        return False
    
    def call_improvement_RMA(self, loss):
        EMA_decay = 0.9
        window_size = 5
        RMA_Threshold = 0.01
        if self.ema_counter == 0:
            self.EMA_list.append(loss)
            self.ema_counter += 1
        else:
            self.EMA_list.append(EMA_decay * self.EMA_list[-1] + (1 - EMA_decay) * loss)
            self.ema_counter += 1
        
        if self.ema_counter >= window_size:
            relative_marginal_gain = ( self.EMA_list[len(self.EMA_list) - window_size] - self.EMA_list[len(self.EMA_list) - 1]) / (self.EMA_list[len(self.EMA_list) - window_size] + 1e-10)
            print(f"relative_marginal_gain: {relative_marginal_gain}")
            if relative_marginal_gain > RMA_Threshold:
                self.counter = 0
            else:
                self.counter += 1
                print(f"counter: {self.counter}")
            if self.counter >= self.patience:
                return self._progress_to_next_group()
        return False

    def _reset_monitors(self):
        self.recent_losses.clear()
        self.running_mean = float('inf')
        self.best_running_mean = float('inf')
        self.counter = 0
        self.ema_moving_average = None   # will be set to first loss in new group
        self.ema_counter = 0
        self.EMA_list = []
        

    def call_simple_compare_best(self, loss):
        if loss < self.best_running_mean:
            self.best_running_mean = loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"counter: {self.counter}")
            if self.counter >= self.patience:
                return self._progress_to_next_group()
        return False

    
    def call_ema_moving_average(self, loss):
        """Update EMA moving average with new loss."""

        # Count this loss since last reset and ignore first five completely
        self.ema_counter += 1
        if self.ema_counter <= self.ema_warmup:
            return False

        # Initialize EMA on first recorded step after warmup
        if self.ema_moving_average is None:
            self.ema_moving_average = loss
            return False

        old_ema = self.ema_moving_average
        self.ema_moving_average = self.ema_alpha * loss + (1 - self.ema_alpha) * old_ema
        print(f"ema_moving_average: {self.ema_moving_average}")
    
        if self.ema_moving_average < old_ema:
            self.counter = 0
        else:
            self.counter += 1
            print(f"counter: {self.counter}")
            if self.counter >= self.patience:
                return self._progress_to_next_group()
        return False


    def _progress_to_next_group(self):
        """Progress to the next timestep group and reset monitoring state."""
        # Reset the counter and clear the buffer
        self._reset_monitors()  
        
        if self.start_from_middle:
            return self._progress_alternating()
        else:
            # Original progression logic
            if self.start_from_low:
                # Progress to higher timestep groups (0 → 966)
                self.current_timestep_groups += 1
            else:
                # Progress to lower timestep groups (966 → 0)
                self.current_timestep_groups -= 1
            
            self.trained_groups.add(self.current_timestep_groups)
            return True

    def _progress_alternating(self):
        """Progress alternately to left and right of the middle starting point."""
        # Find next untrained group alternating between left and right
        if self.next_direction == 'left':
            # Try to go left (lower index)
            next_group = self._find_next_left_group()
            if next_group is not None:
                self.current_timestep_groups = next_group
                self.trained_groups.add(next_group)
                self.next_direction = 'right'  # Next time go right
                return True
            else:
                # No more left groups, try right
                self.next_direction = 'right'
                return self._progress_alternating()
        else:  # self.next_direction == 'right'
            # Try to go right (higher index)
            next_group = self._find_next_right_group()
            if next_group is not None:
                self.current_timestep_groups = next_group
                self.trained_groups.add(next_group)
                self.next_direction = 'left'  # Next time go left
                return True
            else:
                # No more right groups, try left
                self.next_direction = 'left'
                return self._progress_alternating()

    def _find_next_left_group(self):
        """Find the next untrained group to the left (lower index)."""
        for i in range(self.current_timestep_groups - 1, -1, -1):
            if i not in self.trained_groups and i < self.num_timestep_groups:
                return i
        return None

    def _find_next_right_group(self):
        """Find the next untrained group to the right (higher index)."""
        for i in range(self.current_timestep_groups + 1, self.num_timestep_groups):
            if i not in self.trained_groups:
                return i
        return None

    def get_current_group_range(self):
        """Get the current timestep group range being trained [low_bound, high_bound]."""
        if self.start_from_middle:
            # For middle start, just return current group boundaries
            low_bound = self.training_group_boundaries[self.current_timestep_groups]
            if self.current_timestep_groups == self.num_timestep_groups - 1:
                high_bound = 1000
            else:
                high_bound = self.training_group_boundaries[self.current_timestep_groups + 1]
            return [low_bound, high_bound]
            
        elif self.start_from_low:
            # Forward progression: current group boundaries
            low_bound = self.training_group_boundaries[self.current_timestep_groups]
            if self.current_timestep_groups == self.num_timestep_groups - 1:
                high_bound = 1000
            else:
                high_bound = self.training_group_boundaries[self.current_timestep_groups + 1]
        else:
            # Backward progression: current group boundaries
            if self.current_timestep_groups == self.num_timestep_groups:
                high_bound = 1000
            else:
                high_bound = self.training_group_boundaries[self.current_timestep_groups]
            low_bound = self.training_group_boundaries[self.current_timestep_groups - 1]
        
        return [low_bound, high_bound]
    
    def get_timestep_group_gradual(self):
        curr_boundaries = self.get_current_group_range()
        if self.last_boundary is None:
            self.last_boundary = curr_boundaries
            return curr_boundaries
        else:
            if curr_boundaries[0] <= self.last_boundary[0] and curr_boundaries[1] <= self.last_boundary[1]:
                self.last_boundary = [self.last_boundary[0]-1, self.last_boundary[1]-1]
                return self.last_boundary
            else:
                return curr_boundaries

    def get_trained_timesteps_boundaries(self):
        """Get the boundaries of all timesteps that have already been trained [low_bound, high_bound]."""
        if self.start_from_middle:
            # For middle start, return union of all trained group ranges excluding current
            completed_groups = self.trained_groups - {self.current_timestep_groups}
            if not completed_groups:
                return [0, 0]  # No groups completed yet
            
            min_trained = min(completed_groups)
            max_trained = max(completed_groups)
            
            # Check if all groups between min and max are trained (contiguous)
            expected_groups = set(range(min_trained, max_trained + 1))
            if completed_groups == expected_groups:
                # Contiguous range
                low_bound = self.training_group_boundaries[min_trained]
                if max_trained == self.num_timestep_groups - 1:
                    high_bound = 1000
                else:
                    high_bound = self.training_group_boundaries[max_trained + 1]
                return [low_bound, high_bound]
            else:
                # Non-contiguous, return multiple ranges or the full span
                # For simplicity, return the full span from min to max
                low_bound = self.training_group_boundaries[min_trained]
                if max_trained == self.num_timestep_groups - 1:
                    high_bound = 1000
                else:
                    high_bound = self.training_group_boundaries[max_trained + 1]
                return [low_bound, high_bound]
                
        elif self.start_from_low:
            # Forward progression: everything from 0 to current low bound (excluding current)
            if self.current_timestep_groups == 0:
                return [0, 0]  # No groups completed yet
            current_low = self.training_group_boundaries[self.current_timestep_groups]
            return [0, current_low]
        else:
            # Backward progression: everything from current high bound to 1000 (excluding current)
            if self.current_timestep_groups == self.num_timestep_groups - 1:
                return [0, 0]  # No groups completed yet
            if self.current_timestep_groups == self.num_timestep_groups:
                current_high = 1000
            else:
                current_high = self.training_group_boundaries[self.current_timestep_groups]
            return [current_high, 1000]

    def get_if_curriculum_learning_is_done(self):
        """Check if curriculum learning has completed all timestep groups."""
        if self.start_from_middle:
            # Check if all groups have been trained and we're on the last group
            all_groups = set(range(self.num_timestep_groups))
            return (self.trained_groups == all_groups and 
                    self.current_timestep_groups == self.num_timestep_groups - 1)
        elif self.start_from_low:
            return self.current_timestep_groups == self.num_timestep_groups - 1
        else:
            return self.current_timestep_groups == 0