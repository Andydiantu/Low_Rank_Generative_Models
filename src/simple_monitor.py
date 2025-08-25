import math
import torch

class SimpleMonitor:
    def __init__(self, patience=5, start=844, step_size=50, if_start_low=False, start_alpha=0.05, end_alpha=0.5, total_epochs=100):
        self.patience = patience
        self.counter = 0
        self.start = start
        self.step_size = step_size

        self.if_start_low = if_start_low
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.total_epochs = total_epochs

        print(f"Simple monitor initialized with start: {self.start}")
        print(f"Simple monitor initialized with counter: {self.counter}")
        print(f"Simple monitor initialized with patience: {self.patience}")
        print(f"Simple monitor initialized with step size: {self.step_size}")
        print(f"Simple monitor initialized with start_alpha: {self.start_alpha}")
        print(f"Simple monitor initialized with end_alpha: {self.end_alpha}")
        print(f"Simple monitor initialized with total_epochs: {self.total_epochs}")
        print(f"Simple monitor initialized with if_start_low: {self.if_start_low}")


    def __call__(self):
        if self.counter >= self.patience:
            self.counter = 0
            if self.if_start_low:
                self.start += self.step_size
            else:
                self.start -= self.step_size
            return True
        else:
            self.counter += 1
            return False
        
    def get_start(self):
        return self.start
    
    def get_if_curriculum_is_done(self):
        if self.if_start_low:   
            if self.start >= 1000:
                return True
            else:
                return False
        else:
            if self.start < 0:
                return True
            else:
                return False
        
    def cosine_schedule_increasing(self, current_epoch):
        """
        Cosine schedule that INCREASES alpha_e from start_alpha to end_alpha over total_epochs.
        
        Args:
            current_epoch: Current epoch (0-indexed)
        
        Returns:
            alpha_e value for current epoch
        """
        if current_epoch >= self.total_epochs:
            return self.end_alpha
        
        # Modified cosine formula for increasing schedule
        progress = current_epoch / self.total_epochs
        alpha_e = self.start_alpha + (self.end_alpha - self.start_alpha) * (1 - math.cos(math.pi * progress)) / 2
        
        return alpha_e
    

    def cosine_schedule_bell_curve(self, current_epoch):
        """
        Cosine schedule that INCREASES from start_alpha to end_alpha, then DECREASES back to start_alpha.
        Creates a bell curve / inverted U-shape over total_epochs.
        
        Args:
            start_alpha: Initial and final value of alpha_e
            end_alpha: Peak value of alpha_e (at middle of training)
            current_epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
        
        Returns:
            alpha_e value for current epoch
        """
        if current_epoch >= self.total_epochs:
            return self.start_alpha
        
        # Use sine function for bell curve: starts at 0, peaks at π/2, ends at π
        progress = current_epoch / self.total_epochs
        # sin(π * progress) gives us 0 → 1 → 0 over [0, 1]
        alpha_e = self.start_alpha + (self.end_alpha - self.start_alpha) * math.sin(math.pi * progress)
        
        return alpha_e


def importance_weight_schedule( alpha_e, c, T, timesteps, device="cpu", if_start_low=False):
    q = torch.full((T,), 0.0, device=device)
    if if_start_low:
        if c < T:
            q[:c] = (1.0 - alpha_e) / c
            q[c:] = alpha_e / (T - c)
        else:
            q[:] = 1.0 / T  # once fully covered, you can just go uniform
    else:
        if c > 0:
            q[:c] = alpha_e / c
            q[c:] = (1.0 - alpha_e) / (T - c)
        else:
            q[:] = alpha_e / T

    w_iw = (1.0 / T) / q[timesteps - 1]  
    w_iw = torch.clamp(w_iw, max=5.0)
    w_iw = w_iw / w_iw.mean()


    return w_iw


def sample_timesteps_curriculum(bs, alpha_e, c, T, device="cpu", reverse_curriculum=False):
    """
    Sample timesteps with curriculum learning strategy.
    
    Args:
        bs: batch size
        alpha_e: probability weight for curriculum mixing
        c: cutoff timestep 
        T: total timesteps
        device: torch device
        reverse_curriculum: if True, c becomes lower bound (reverse curriculum)
                          if False, c becomes upper bound (normal curriculum)
    
    Returns:
        t: sampled timesteps
    """
    coin = torch.rand(bs, device=device)
    t = torch.empty(bs, dtype=torch.long, device=device)
    
    if not reverse_curriculum:
        # Normal curriculum: c is upper bound
        # (1-α_e) samples from [1..c] (easier), α_e samples from [c+1..T] (harder)
        idx_easy = coin >= alpha_e  # (1-α_e) probability
        idx_hard = coin < alpha_e   # α_e probability
        
        t[idx_easy] = torch.randint(1, c+1, (idx_easy.sum(),), device=device)
        if c < T:
            t[idx_hard] = torch.randint(c+1, T, (idx_hard.sum(),), device=device)
        else:
            t[idx_hard] = torch.randint(int(alpha_e*T), T, (idx_hard.sum(),), device=device)
    else:
        # Reverse curriculum: c is lower bound  
        # (1-α_e) samples from [c+1..T] (harder), α_e samples from [1..c] (easier)
        idx_hard = coin >= alpha_e  # (1-α_e) probability  
        idx_easy = coin < alpha_e   # α_e probability
        
        if c < T:
            t[idx_hard] = torch.randint(c+1, T, (idx_hard.sum(),), device=device)
        else:
            t[idx_hard] = torch.randint(int((1-alpha_e)*T), T, (idx_hard.sum(),), device=device)
        t[idx_easy] = torch.randint(1, c+1, (idx_easy.sum(),), device=device)
    
    return t
        