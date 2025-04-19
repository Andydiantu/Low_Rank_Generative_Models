# VRAM Usage Tracker

A simple utility to track and visualize GPU VRAM usage during the execution of deep learning scripts.

## Features

- Track VRAM usage of any Python script or function
- Generate a line plot showing VRAM usage over time
- Report maximum VRAM usage
- Support for tracking usage on specific CUDA devices
- Configurable sampling interval

## Usage

### Command Line

Track VRAM usage of a Python script:

```bash
python src/vram_tracker.py path/to/your_script.py --interval 0.5 --output vram_plot.png
```

Arguments:
- `script`: Path to the Python script you want to track
- `--interval`: (Optional) Sampling interval in seconds (default: 0.5)
- `--output`: (Optional) Output path for the plot (default: vram_usage_TIMESTAMP.png)

### Python API

You can also use the tracker in your Python code:

```python
from src.vram_tracker import VRAMTracker, track_function

# Option 1: Track a specific function
def my_training_function(model, data):
    # ... training code ...
    return model

result, max_vram = track_function(
    my_training_function, 
    model, 
    data,
    interval=0.5,
    plot=True,
    save_path="vram_plot.png"
)

# Option 2: Track a code block
tracker = VRAMTracker(device="cuda:0", interval=0.5)
tracker.start()

# Your code here
model = ...
model.train()
# ...

tracker.stop()
max_usage = tracker.plot(save_path="vram_plot.png")
print(f"Maximum VRAM usage: {max_usage:.2f} MB")
```

## Requirements

- PyTorch
- matplotlib
- CUDA-enabled GPU 