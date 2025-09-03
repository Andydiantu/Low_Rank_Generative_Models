#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=aw624 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc

# Activate virtualenv if present; otherwise fall back to conda or system python
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "No venv found at venv/bin/activate; continuing with current Python"
fi

# Source CUDA setup if present
if [ -f "/vol/cuda/12.0.0/setup.sh" ]; then
  source /vol/cuda/12.0.0/setup.sh
else
  echo "No CUDA setup at /vol/cuda/12.0.0/setup.sh; relying on driver-provided CUDA"
fi
/usr/bin/nvidia-smi
uptime


# timestamp=$(date +"%Y%m%d_%H%M%S")
# mkdir -p tmux_log/$(date +%m%d)
# python -u src/encode_imagenet_128.py 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt

timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)

# Use fewer dataloader workers in SLURM/Jupyter to avoid silent hangs
export DATALOADER_WORKERS=${DATALOADER_WORKERS:-2}

python -u src/DiT_trainer.py --set num_epochs=1200 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt
